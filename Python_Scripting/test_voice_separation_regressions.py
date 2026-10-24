"""Regression tests for routing. Audio models are stubbed, not quality-tested."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import voice_routing as routing
import extract_voices as extract


class RoutingTests(unittest.TestCase):
    def test_single_male_does_not_force_a_female_group(self):
        y=np.ones(48000,dtype=np.float32)*.1
        encoder=types.SimpleNamespace(embed=lambda *a,**kw:np.array([1.,0.]))
        estimate=lambda *a:(np.full(100,125.),np.ones(100,bool),1.,160)
        regions=routing.embedding_regions(y,16000,[(0,3)],encoder,estimate)
        self.assertEqual({r.label for r in regions},{'male'})

    def test_two_female_speakers_do_not_invent_a_male_label(self):
        y=np.r_[np.ones(32000),np.full(32000,2)].astype(np.float32)
        encoder=types.SimpleNamespace(embed=lambda x,**kw:np.array([1.,0.]) if np.mean(x)<1.5 else np.array([0.,1.]))
        estimate=lambda x,sr:(np.full(100,195. if np.mean(x)<1.5 else 260.),np.ones(100,bool),1.,160)
        regions=routing.embedding_regions(y,16000,[(0,2),(2,4)],encoder,estimate)
        self.assertEqual({r.label for r in regions},{'female'})
        self.assertEqual(len({r.speaker for r in regions}),2)

    def test_reference_labels_override_pitch_assumptions(self):
        class Encoder:
            def embed(self,audio,sr=None):
                if isinstance(audio,str): return np.array([1.,0.]) if audio=='female.wav' else np.array([0.,1.])
                return np.array([1.,0.]) if np.mean(audio)<1.5 else np.array([0.,1.])
        y=np.r_[np.ones(32000),np.full(32000,2)].astype(np.float32)
        estimate=lambda *args:(_ for _ in ()).throw(AssertionError('Reference matching must not use pitch'))
        regions=routing.embedding_regions(y,16000,[(0,2),(2,4)],Encoder(),estimate,
                                           {'female':'female.wav','male':'male.wav'})
        self.assertEqual([r.label for r in regions[:4]],['female']*4)
        self.assertEqual([r.label for r in regions[4:]],['male']*4)

    def test_ambiguous_mixture_is_not_forced_into_a_gender(self):
        encoder=types.SimpleNamespace(embed=lambda path:np.array([1.,0.]) if path=='f' else np.array([0.,1.]))
        matcher=routing.ReferenceMatcher(encoder,{'female':'f','male':'m'})
        self.assertEqual(matcher.match_embedding([1,1])[0],'unknown')
        self.assertEqual(matcher.match_embedding([-1,-1])[0],'unknown')

    def test_one_reference_never_infers_the_other_gender(self):
        matcher=routing.ReferenceMatcher(types.SimpleNamespace(embed=lambda path:np.array([1.,0.])),{'female':'f'})
        self.assertEqual(matcher.match_embedding([0,1])[0],'unknown')

    def test_same_reference_embedding_is_rejected(self):
        with self.assertRaises(ValueError):
            routing.ReferenceMatcher(types.SimpleNamespace(embed=lambda path:np.array([1.,0.])),{'female':'f','male':'m'})

    def test_no_pause_speaker_change_is_split_on_pitch_track(self):
        y=np.ones(32000,dtype=np.float32)*.1
        estimate=lambda *a:(np.r_[np.full(100,125.),np.full(100,230.)],np.ones(200,bool),1.,160)
        regions=routing.pitch_regions(y,16000,[(0,2)],estimate)
        self.assertEqual(regions[0].label,'male')
        self.assertEqual(regions[-1].label,'female')
        self.assertNotIn('overlap',{r.label for r in regions})
        self.assertAlmostEqual(sum(r.end-r.start for r in regions),2.)

    def test_ambiguous_pitch_and_unvoiced_audio_go_to_review(self):
        self.assertEqual(routing.pitch_label([163,165,166])[0],'unknown')
        self.assertEqual(routing.pitch_label([])[0],'unknown')
        self.assertEqual(routing.pitch_regions(np.zeros(32000),16000,[(0,2)],None)[0].label,'unknown')

    def test_isolated_octave_error_does_not_split_speaker(self):
        f0=np.full(100,120.);f0[50]=240.
        estimate=lambda *a:(f0,np.ones(100,bool),1.,160)
        regions=routing.pitch_regions(np.ones(16000)*.1,16000,[(0,1)],estimate)
        self.assertEqual({r.label for r in regions},{'male'})

    def test_diarization_overlap_is_excluded_from_both_solo_outputs(self):
        regions=routing.partition_turns([(0,2,'a'),(1,3,'b')],{'a':'female','b':'male'})
        self.assertEqual([(r.start,r.end,r.label) for r in regions],[(0,1,'female'),(1,2,'overlap'),(2,3,'male')])
        y=np.ones(3000,dtype=np.float32)
        female=routing.assemble(y,1000,regions,{'female'},fade_ms=0)
        male=routing.assemble(y,1000,regions,{'male'},fade_ms=0)
        self.assertEqual(np.count_nonzero(female[1000:]),0)
        self.assertEqual(np.count_nonzero(male[:2000]),0)

    def test_duplicate_turns_from_same_speaker_do_not_duplicate_samples(self):
        regions=routing.partition_turns([(0,2,'a'),(1,3,'a')],{'a':'female'})
        audio=routing.assemble(np.ones(3000),1000,regions,{'female'},silence=0,fade_ms=0)
        self.assertEqual(len(audio),3000)
        self.assertEqual(set(r.label for r in regions),{'female'})

    def test_auto_output_preserves_timeline_and_concat_preserves_sample_count(self):
        regions=[routing.Region(.5,1,'female'),routing.Region(1,1.5,'female'),routing.Region(2,2.5,'female')]
        y=np.arange(3000,dtype=np.float32)/3000
        auto=routing.assemble(y,1000,regions,{'female'},minimum=.8,fade_ms=0)
        self.assertEqual(len(auto),3000)
        np.testing.assert_array_equal(auto[500:1500],y[500:1500])
        self.assertEqual(np.count_nonzero(auto[1500:]),0)
        joined=routing.assemble(y,1000,regions,{'female'},silence=.1,fade_ms=0)
        self.assertEqual(len(joined),1600)

    def test_analysis_cells_cover_speech_once(self):
        cells,contexts=routing.analysis_cells([(0,2.3),(3,3.2)])
        self.assertAlmostEqual(sum(b-a for a,b in cells),2.5)
        self.assertTrue(all(a<b for a,b in contexts))
        self.assertTrue(all(b-a<=1.5+1e-6 for a,b in contexts))

    def test_energy_detection_keeps_final_samples_and_silence_is_empty(self):
        self.assertEqual(extract.detect_segments(np.zeros(32000),16000),[])
        regions=extract.detect_segments(np.ones(32001)*.1,16000)
        self.assertEqual(regions,[(0,32001/16000)])


class ProcessingTests(unittest.TestCase):
    def test_music_removal_does_not_bypass_requested_voice_routing(self):
        captured=[]
        with patch.object(extract,'remove_music_demucs',return_value=(np.ones(16000,dtype=np.float32)*.1,16000)), \
             patch.object(extract,'detect_segments',return_value=[(0,1)]), \
             patch.object(extract,'_route',return_value=[routing.Region(0,1,'female')]) as route, \
             patch.object(extract,'_export_regions',side_effect=lambda *a,**kw:captured.append(a)), \
             patch.object(extract,'save_audio') as direct:
            extract.process('source','out.wav',remove_music=True,split_output=True)
        route.assert_called_once()
        direct.assert_not_called()
        self.assertTrue(captured[0][7])

    def test_music_removal_failure_is_not_silently_ignored(self):
        with patch.object(extract,'remove_music_demucs',return_value=(None,None)):
            with self.assertRaises(RuntimeError): extract._load_processed_audio('source',remove_music=True)

    def test_pitch_ambiguity_never_routes_to_male_fallback(self):
        with patch.object(extract,'_f0_pyin',return_value=(np.full(100,165.),np.ones(100,bool),1.,160)):
            self.assertEqual(extract.classify(np.ones(16000)*.1,16000,0,1)[0],'unknown')
            self.assertIsNone(extract._pick_source(np.ones(16000)*.1,np.ones(16000)*.1,16000,'male_solo',165))

    def test_sepformer_processes_overlap_once_for_both_outputs(self):
        saved={}
        y=np.ones(32000,dtype=np.float32)*.1
        sources=(np.ones_like(y)*.03,np.ones_like(y)*.04)
        labels=[('female',.8,'reference_match'),('male',.8,'reference_match')]
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(extract,'_load_processed_audio',return_value=(y,16000)), \
             patch.object(extract,'_load_sepformer',return_value=object()), \
             patch.object(extract,'detect_segments',return_value=[(0,2)]), \
             patch.object(extract,'_sepformer_separate',return_value=sources) as separate, \
             patch.object(extract,'_source_labels',return_value=labels), \
             patch.object(extract,'classify',side_effect=AssertionError('Do not reject mixtures before separation')), \
             patch.object(extract,'save_audio',side_effect=lambda audio,sr,path,*args:saved.setdefault(Path(path).name,audio)), \
             contextlib.redirect_stdout(io.StringIO()):
            extract.process_sepformer('input',str(Path(tmp)/'out.wav'),split_output=True)
            self.assertTrue((Path(tmp)/'out_separation.json').exists())
        self.assertEqual(separate.call_count,1)
        self.assertEqual(set(saved),{'out_female.wav','out_male.wav'})
        self.assertAlmostEqual(float(saved['out_female.wav'][16000]),.03,places=6)
        self.assertAlmostEqual(float(saved['out_male.wav'][16000]),.04,places=6)

    def test_combined_separated_sources_have_common_headroom(self):
        y=np.ones(16000,dtype=np.float32)*.1
        saved=[]
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(extract,'_load_processed_audio',return_value=(y,16000)), \
             patch.object(extract,'_load_sepformer',return_value=object()), \
             patch.object(extract,'detect_segments',return_value=[(0,1)]), \
             patch.object(extract,'_sepformer_separate',return_value=(np.full_like(y,.8),np.full_like(y,.8))), \
             patch.object(extract,'_source_labels',return_value=[('female',.8,'reference_match'),('male',.8,'reference_match')]), \
             patch.object(extract,'save_audio',side_effect=lambda audio,*args:saved.append(audio)), \
             contextlib.redirect_stdout(io.StringIO()):
            extract.process_sepformer('in',str(Path(tmp)/'out.wav'),keep_set={'female_solo','male_solo'})
        self.assertEqual(len(saved),1)
        self.assertLessEqual(float(np.max(np.abs(saved[0]))),.980001)
        self.assertGreater(float(np.max(np.abs(saved[0]))),.97)

    def test_sepformer_failure_does_not_export_unseparated_mix(self):
        with patch.object(extract,'_load_processed_audio',return_value=(np.ones(16000)*.1,16000)), \
             patch.object(extract,'_load_sepformer',side_effect=RuntimeError('Model unavailable')), \
             patch.object(extract,'save_audio') as save,contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError): extract.process_sepformer('in','out.wav')
        save.assert_not_called()

    def test_split_review_and_json_preserve_ambiguous_audio(self):
        saved={}
        regions=[routing.Region(0,1,'female'),routing.Region(1,2,'unknown'),routing.Region(2,3,'male')]
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(extract,'save_audio',side_effect=lambda audio,sr,path,*args:saved.setdefault(Path(path).name,audio)), \
             contextlib.redirect_stdout(io.StringIO()):
            extract._export_regions(np.ones(3000)*.1,1000,regions,str(Path(tmp)/'out.wav'),{'female_solo'},'auto',True)
            report=json.loads((Path(tmp)/'out_separation.json').read_text())
        self.assertEqual(set(saved),{'out_female.wav','out_male.wav','out_review.wav'})
        self.assertEqual(report['durations']['unknown'],1.)
        self.assertEqual(np.count_nonzero(saved['out_review.wav'][:1000]),0)
        self.assertGreater(np.count_nonzero(saved['out_review.wav'][1000:2000]),900)


class IntegrationTests(unittest.TestCase):
    def test_gui_keeps_target_when_music_removed_and_passes_references(self):
        from test_clonage_regressions import FakeWidget, extract_functions
        FakeWidget.entries={};FakeWidget.labels={}
        checkboxes,callbacks,commands={},{},[]
        class Variable:
            def __init__(self,value='',**kwargs): self.value=value;self.callbacks=[]
            def get(self): return self.value
            def set(self,value):
                self.value=value
                for callback in self.callbacks: callback()
            def trace_add(self,mode,callback): self.callbacks.append(callback)
        class Widget(FakeWidget):
            def __init__(self,parent=None,**kwargs):
                super().__init__(parent,**kwargs)
                if 'variable' in kwargs: checkboxes[kwargs.get('text','')]=kwargs['variable']
        tk=types.SimpleNamespace(**{key:Widget for key in ('Frame','Label','Entry','Checkbutton')},
                                  StringVar=Variable,BooleanVar=Variable)
        def add_row(parent,label,var,*args,**kwargs): FakeWidget.entries[label]=var
        def make_btn(parent,text,callback,row): callbacks[text]=callback;return Widget(),Widget()
        namespace=dict(tk=tk,ttk=types.SimpleNamespace(Combobox=Widget),os=os,sys=sys,
                       DIR_VOICES='/voices',DIR_OUTPUT='/output',SCRIPTS_DIR='/scripts',
                       scrollable_tab=lambda *args:(Widget(),Widget()),add_row=add_row,
                       add_console=lambda *args:Widget(),make_btn=make_btn,
                       run_cmd=lambda command,*args:commands.append(command),log=lambda *args:None)
        extract_functions('xtts_studio.py',{'tab_extract'},namespace)['tab_extract'](None)
        FakeWidget.entries['Audio/Video source'].set('/input.wav')
        FakeWidget.entries['Output (wav/mp3)'].set('/output.wav')
        FakeWidget.entries['Female solo reference'].set('/female solo.wav')
        FakeWidget.entries['Male solo reference'].set('/male solo.wav')
        self.assertEqual(FakeWidget.entries['Method'].get(),'ecapa')
        checkboxes['Remove background music (demucs)'].set(True)
        self.assertEqual(FakeWidget.entries['Keep'].get(),'female')
        checkboxes['Split F+M  (writes _female + _male)'].set(True)
        callbacks['>  Separate'](None)
        command=commands[0]
        self.assertEqual(command[command.index('--female-ref')+1],'/female solo.wav')
        self.assertEqual(command[command.index('--male-ref')+1],'/male solo.wav')
        self.assertIn('--remove-music',command)
        self.assertIn('--split-output',command)
        self.assertIn('--match-minimum',command)
        self.assertNotIn('--overlap-range',command)

    def test_pyannote_new_and_legacy_api_exclude_mixed_intervals(self):
        turns=[(0,2,'a'),(1,3,'b')]
        class Annotation:
            def itertracks(self,**kwargs):
                for start,end,speaker in turns:
                    yield types.SimpleNamespace(start=start,end=end),None,speaker
        for version in ('new','legacy'):
            with self.subTest(version=version):
                calls,exported=[],[]
                class Instance:
                    def to(self,device): calls.append(('device',device))
                    def __call__(self,payload):
                        calls.append(('shape',payload['waveform'].shape))
                        return types.SimpleNamespace(speaker_diarization=Annotation()) if version=='new' else Annotation()
                class Pipeline:
                    @staticmethod
                    def from_pretrained(model,token=None): calls.append(('token',token));return Instance()
                if version=='legacy':
                    def legacy(model,use_auth_token=None):
                        calls.append(('token',use_auth_token));return Instance()
                    Pipeline.from_pretrained=staticmethod(legacy)
                torch=types.SimpleNamespace(from_numpy=lambda x:x,device=lambda name:name)
                modules={'torch':torch,'pyannote.audio':types.SimpleNamespace(Pipeline=Pipeline)}
                y=np.r_[np.ones(16000)*.1,np.ones(16000)*.15,np.ones(16000)*.2].astype(np.float32)
                def estimate(chunk,sr,device):
                    median=230. if chunk.mean()<.15 else 120.
                    return np.full(100,median),np.ones(100,bool),1.,160
                with patch.dict(sys.modules,modules),patch.object(extract,'_load_processed_audio',return_value=(y,16000)), \
                     patch.object(extract,'_estimate_pitch',side_effect=estimate), \
                     patch.object(extract,'_export_regions',side_effect=lambda *args:exported.append(args)), \
                     contextlib.redirect_stdout(io.StringIO()):
                    extract.process_pyannote('in','out.wav',{'female_solo'},'auto','none',165,'test-token')
                regions=exported[0][2]
                self.assertEqual([(r.start,r.end,r.label) for r in regions],[(0,1,'female'),(1,2,'overlap'),(2,3,'male')])
                self.assertIn(('token','test-token'),calls)
                self.assertIn(('shape',(1,48000)),calls)

    def test_report_uses_union_duration_for_two_same_label_sources(self):
        regions=[routing.Region(0,2,'female'),routing.Region(1,3,'female')]
        with tempfile.TemporaryDirectory() as tmp:
            report=routing.write_report(str(Path(tmp)/'report.json'),regions,'sepformer')
        self.assertEqual(report['durations']['female'],3.)

    def test_missing_new_audio_does_not_leave_old_stem_unannounced(self):
        with tempfile.TemporaryDirectory() as tmp:
            old=Path(tmp)/'out_male.wav';old.write_bytes(b'previous result')
            with patch.object(extract,'save_audio'),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(FileExistsError):
                    extract._export_regions(np.ones(1000),1000,[],str(Path(tmp)/'out.wav'),{'female_solo'},'auto',True)
            self.assertEqual(old.read_bytes(),b'previous result')


class CliTests(unittest.TestCase):
    def test_cli_split_routes_music_and_speaker_references_to_ecapa(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.wav';female=Path(tmp)/'solo f.wav';male=Path(tmp)/'solo m.wav'
            for path in (source,female,male):path.touch()
            argv=['extract_voices.py',str(source),str(Path(tmp)/'out.wav'),'--method','ecapa',
                  '--female-ref',str(female),'--male-ref',str(male),'--remove-music','--split-output']
            with patch.object(sys,'argv',argv),patch.object(extract,'sf',object()),patch.object(extract,'librosa',object()), \
                 patch.object(extract,'process') as process:
                extract.main()
            self.assertEqual(process.call_args.kwargs['cluster_method'],'ecapa')
            self.assertEqual(process.call_args.kwargs['female_ref'],str(female))
            self.assertEqual(process.call_args.kwargs['male_ref'],str(male))
            self.assertTrue(process.call_args.args[9])
            self.assertTrue(process.call_args.args[15])

    def test_invalid_silence_is_rejected(self):
        for value in ('nan','inf','-1'):
            with self.subTest(value=value),self.assertRaises(Exception):extract.parse_silence(value)

    def test_temporary_video_audio_removed_if_separator_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            video=Path(tmp)/'source.mp4';video.touch();used=[]
            def ffmpeg(command,**kwargs): used.append(command[-1])
            argv=['extract_voices.py',str(video),str(Path(tmp)/'out.wav'),'--method','sepformer']
            with patch.object(sys,'argv',argv),patch.object(extract,'sf',object()),patch.object(extract,'librosa',object()), \
                 patch.object(extract.subprocess,'run',side_effect=ffmpeg), \
                 patch.object(extract,'process_sepformer',side_effect=RuntimeError('failed')), \
                 contextlib.redirect_stdout(io.StringIO()),self.assertRaises(RuntimeError):
                extract.main()
            self.assertEqual(len(used),1)
            self.assertFalse(os.path.exists(used[0]))


if __name__=='__main__':
    unittest.main()
