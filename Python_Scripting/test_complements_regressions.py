"""Offline regressions for reference preparation, DSP, diagnostics and utilities.

python3 -m unittest -v test_clonage_regressions test_complements_regressions
No XTTS, GPU, Whisper or ECAPA models are loaded by these tests.
"""
import contextlib
import importlib
import io
import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import wave
import numpy as np
from scipy.signal import lfilter

import audio_checks
import curate_reference as curate
import ltas_match as ltas
import speaker_identity as identity
import text_alignment as alignment
import apply_pitch_to_clone as pitch
import test_truncation as truncation
import test_max_ref_len as cap_test
import prepare_rvc_dataset as dataset
import brainwave_studio as brain
import xtts_settings as settings
import pron_score
import listening_test


class ReferenceTests(unittest.TestCase):
    def test_silence_and_low_noise_not_voiced(self):
        self.assertEqual(curate._voiced_fraction(np.zeros(16000)), 0.)
        self.assertEqual(curate._voiced_fraction(np.full(16000, 1e-8)), 0.)
        self.assertGreater(curate._voiced_fraction(np.sin(np.arange(16000)*.1)*.1), .99)

    def test_invalid_hop_fails_before_encoder_import(self):
        for hop in (0, -1, np.nan, 5):
            with self.assertRaises(ValueError):
                curate.curate(['nonexistent.wav'], 'output.wav', hop=hop)

    def test_audio_checks_and_downmix(self):
        for audio in ([], [np.nan], [np.inf], np.zeros(50)):
            with self.assertRaises(ValueError):
                audio_checks.mono_audio(audio)
        np.testing.assert_allclose(audio_checks.mono_audio([[1., 0.], [0., 1.]]), [.5, .5])
        with self.assertRaises(ValueError):
            identity._resample(np.ones(10), 0, 16000)

    def test_curation_caps_output_and_does_not_cross_sources(self):
        class Encoder:
            def __init__(self, **kwargs): pass
            def embed(self, y, sr): return np.array([1., 0.])
        written = []
        source = np.sin(np.arange(16000*6)*.05).astype(np.float32)*.1
        fake_sf = types.SimpleNamespace(write=lambda path,y,sr:written.append((y,sr)))
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp)/'a.wav', Path(tmp)/'b.wav']
            for path in paths: path.touch()
            with patch.dict(sys.modules, {'speaker_identity':types.SimpleNamespace(SpeakerEncoder=Encoder), 'soundfile':fake_sf}), \
                 patch.object(curate, '_load_mono', return_value=(source,16000)), \
                 contextlib.redirect_stdout(io.StringIO()):
                curate.curate(paths, str(Path(tmp)/'out.wav'), keep_seconds=5, window=4, hop=2)
            self.assertEqual(len(written[0][0]), 80000)
            self.assertEqual(written[0][1], 16000)


class SpectrumTests(unittest.TestCase):
    def test_exact_known_bell_is_not_recentered(self):
        grid = np.geomspace(80,8000,256)
        ref = ltas.eq_peaking_db(grid,24000,[4,0,0]); ref -= ref.max()
        result = ltas.fit_eq_ls(grid,ref,np.zeros_like(ref))
        self.assertAlmostEqual(result['eq_low'],4.,places=1)
        self.assertEqual(result['eq_mid'],0.)
        self.assertEqual(result['eq_high'],0.)
        self.assertLessEqual(result['residual_after'],.01)

    def test_residual_matches_rounded_exported_gains(self):
        grid = np.geomspace(80,8000,256)
        ref = ltas.eq_peaking_db(grid,24000,[2.26,-1.37,3.44])
        result = ltas.fit_eq_ls(grid,ref,np.zeros_like(ref),weighting='voice')
        gains=[result[k] for k,_,_ in ltas.EQ_BANDS]
        residual=ltas.eq_peaking_db(grid,24000,gains)-ref
        weights=np.maximum(10**(ref/20)/max(10**(ref/20)),.1)
        residual-=np.average(residual,weights=weights)
        self.assertEqual(result['residual_after'],round(float(np.sqrt(np.average(residual**2,weights=weights))),2))
        self.assertTrue(all(-6 <= g <= 6 for g in gains))

    def test_fit_large_offset_and_invalid_data(self):
        grid=np.geomspace(80,8000,256)
        result=ltas.fit_eq_ls(grid,np.full(256,100.),np.zeros(256))
        self.assertTrue(np.isfinite(result['residual_after']))
        for audio in ([],np.zeros(1000),[np.nan]):
            with self.assertRaises(ValueError): ltas.compute_ltas(audio,24000)
        with self.assertRaises(ValueError): ltas.compute_ltas(np.ones(1000),100)
        with self.assertRaises(ValueError): ltas.fit_eq_ls(grid,[1],np.zeros(256))

    def test_upper_band_is_really_measured(self):
        grid=np.geomspace(80,8000,256); ref=np.zeros_like(grid); clone=ref.copy()
        clone[grid>=6000]=10
        _,lp=ltas.derive_hp_lp(grid,ref,clone)
        self.assertGreater(lp,0)
        self.assertGreater(sum(grid>=6000),10)

    @unittest.skipUnless(__import__('shutil').which('ffmpeg'),'ffmpeg not installed')
    def test_bell_response_matches_actual_ffmpeg(self):
        sr=24000; t=np.arange(sr*2)/sr
        for freq in (120,1500,5000):
            audio=(.03*np.sin(2*np.pi*freq*t)).astype('<f4')
            proc=subprocess.run(['ffmpeg','-v','error','-f','f32le','-ar',str(sr),'-ac','1','-i','pipe:0',
                                 '-af','equalizer=f=120:t=h:w=160:g=4','-f','f32le','pipe:1'],
                                 input=audio.tobytes(),capture_output=True,check=True)
            out=np.frombuffer(proc.stdout,dtype='<f4')
            measured=20*np.log10(np.sqrt(np.mean(out[sr:]**2))/np.sqrt(np.mean(audio[sr:]**2)))
            expected=ltas.rbj_peaking_db(np.array([freq]),sr,120,160,4)[0]
            self.assertAlmostEqual(measured,expected,places=2)


class EncoderTests(unittest.TestCase):
    def test_device_and_savedir_cache(self):
        torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:False))
        with patch.dict(sys.modules,{'torch':torch}),patch.object(identity.SpeakerEncoder,'_models',{}), \
             patch.object(identity.SpeakerEncoder,'_build',lambda self:object()):
            a=identity.SpeakerEncoder('cpu','/tmp/cache_a')
            b=identity.SpeakerEncoder('cuda','/tmp/cache_a')
            c=identity.SpeakerEncoder('cpu','/tmp/cache_a')
            d=identity.SpeakerEncoder('cpu','/tmp/cache_b')
            self.assertIs(a.model,c.model)
            self.assertIsNot(a.model,b.model)
            self.assertIsNot(a.model,d.model)
            self.assertEqual(b.device,'cuda:0')

    def test_cosine_rejects_invalid_embedding(self):
        for vector in ([0,0],[np.nan,1],[1]):
            with self.assertRaises(ValueError): identity.SpeakerEncoder.cosine([1,0],vector)
        self.assertEqual(identity.SpeakerEncoder.cosine([1,0],[-1,0]),-1)

    def test_silent_split_anchor_unavailable(self):
        encoder=identity.SpeakerEncoder.__new__(identity.SpeakerEncoder)
        self.assertIsNone(encoder.self_ceiling(np.zeros(64000)))


class AlignmentTests(unittest.TestCase):
    def test_repeated_intro_does_not_mask_missing_ending(self):
        first='un deux trois quatre cinq six sept huit'.split()
        expected=first+[f'mot{i}' for i in range(24)]+first
        result=alignment.ending_metrics(' '.join(expected[:35]),' '.join(expected))
        self.assertEqual(result['missing_suffix'],5)
        self.assertAlmostEqual(result['tail_kept'],3/8)
        self.assertTrue(result['suspect'])

    def test_complete_and_internal_errors(self):
        expected='Respire calmement et laisse aller tes épaules.'
        self.assertFalse(alignment.ending_metrics(expected,expected)['suspect'])
        self.assertEqual(alignment.ending_metrics('',expected)['tail_kept'],0.)
        with self.assertRaises(ValueError): alignment.ending_metrics('hello','')

    def test_repeated_pitch_marks_align_to_distinct_occurrences(self):
        text,marks=pitch.parse_pitch_tags('calme[p:-2] puis calme[p:+3]')
        result={'segments':[{'words':[
            dict(word='calme',start=0,end=.4),dict(word='puis',start=.5,end=.8),
            dict(word='calme',start=1,end=1.4)]}]}
        edits=pitch.aligned_corrections(text,marks,result,1500)
        self.assertEqual(edits,[(0,400,-2,'calme'),(1000,1400,3,'calme')])

    def test_pitch_offsets_cross_sentences_and_no_partial_match(self):
        sentences=[pitch.parse_pitch_tags('calme[p:-2] ici'),pitch.parse_pitch_tags('puis calme[p:+3]')]
        text,marks=pitch.combine_annotations(sentences)
        self.assertEqual([m['start'] for m in marks],[0,3])
        result={'segments':[{'words':[dict(word='calmement',start=0,end=1)]}]}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pitch.aligned_corrections(text,marks,result,1000),[])

    def test_hyphen_and_apostrophe_tokenization(self):
        text,marks=pitch.parse_pitch_tags("l'air[p:-2] entre-toi[p:+1]")
        self.assertEqual([(m['start'],m['end']) for m in marks],[(0,2),(2,4)])
        self.assertEqual(alignment.words("L’air"),alignment.words("l'air"))

    def test_long_text_wrong_language_requires_custom_text(self):
        with self.assertRaises(ValueError): truncation.load_texts(None,'ja',truncation.LONG_TEXTS)

    def test_reference_cap_experiment_reuses_fixed_gpt(self):
        calls=[]
        def compute(model,ref,**kwargs):
            calls.append(kwargs['max_ref_len'])
            return object(),kwargs['max_ref_len']
        fake=types.SimpleNamespace(compute_latents=compute)
        result=cap_test.isolated_latents(None,'ref',settings.validate({}),[20,30,45],45,fake)
        self.assertIs(result[20][0],result[45][0])
        self.assertIs(result[30][0],result[45][0])
        self.assertEqual(calls,[45,20,30])
        self.assertEqual([result[v][1] for v in result],[20,30,45])


class MixingTests(unittest.TestCase):
    def render(self,segments,**kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            path=str(Path(tmp)/'out.wav')
            toolbox=brain.ToneToolbox(sample_rate=8000,amplitude=.7)
            toolbox.stream_session(segments,path,fade=0,xfade=0,block=1024,**kwargs)
            with wave.open(path,'rb') as reader:
                return np.frombuffer(reader.readframes(reader.getnframes()),dtype='<i2')

    def test_loud_segment_drone_no_clipping(self):
        samples=self.render([dict(mode='binaural',carrier=400,beat=6,duration=2,
                                  drone=.9,noise=0,tone_level=1)])
        self.assertEqual(len(samples),32000)
        self.assertLessEqual(np.max(np.abs(samples)),round(.9*32767)+1)
        self.assertEqual(np.count_nonzero(np.abs(samples)>=32766),0)

    def test_follow_levels_and_global_music_headroom(self):
        samples=self.render([dict(mode='binaural',carrier=400,beat=6,duration=1,
                                  tone_level=3,drone=.9,noise=.6,level_drone=True,level_noise=True)],
                            music=np.full((1000,2),1.2),music_level=.8)
        self.assertLessEqual(np.max(np.abs(samples)),round(.9*32767)+1)

    def test_quiet_tone_is_not_amplified(self):
        samples=self.render([dict(mode='binaural',carrier=400,beat=6,duration=1,tone_level=.1)])
        self.assertLessEqual(np.max(np.abs(samples)),round(.07*32767)+1)

    def test_invalid_segment_fails(self):
        base=dict(mode='isochronic',carrier=400,beat=6,duration=1)
        for updates in (dict(duration=0),dict(beat=float('nan')),dict(drone=-1),dict(duty=0),dict(tone_level=float('inf'))):
            with self.assertRaises(ValueError): brain.validate_segment(dict(base,**updates))
        with self.assertRaises(ValueError): self.render([dict(base,carrier=5000)])


class UtilityTests(unittest.TestCase):
    def test_dataset_split_preserves_last_sample(self):
        fake=types.SimpleNamespace(effects=types.SimpleNamespace(split=lambda y,top_db:np.array([[0,len(y)]])))
        with patch.dict(sys.modules,{'librosa':fake}):
            intervals=dataset.split_utterances(np.ones(10001),1000,min_utt=1,max_utt=4)
        self.assertEqual(intervals[0][0],0)
        self.assertEqual(intervals[-1][1],10001)
        self.assertEqual(sum(e-s for s,e in intervals),10001)
        for a,b in zip(intervals,intervals[1:]): self.assertEqual(a[1],b[0])

    def test_dataset_refuses_old_clips_before_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref=Path(tmp)/'source.wav';ref.touch()
            output=Path(tmp)/'dataset';output.mkdir();(output/'0099.wav').write_bytes(b'preserve')
            with patch.object(sys,'argv',['prepare_rvc_dataset.py',str(ref),'-o',str(output)]), \
                 contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as exc:
                dataset.main()
            self.assertEqual(exc.exception.code,2)
            self.assertEqual((output/'0099.wav').read_bytes(),b'preserve')

    def test_empty_wer_and_unicode_apostrophe(self):
        self.assertEqual(pron_score.word_error_rate('','hallucination'),1.)
        self.assertEqual(pron_score.word_error_rate('',''),0.)
        self.assertEqual(pron_score.word_error_rate('L’air passe',"l'air passe"),0.)

    def test_no_speech_is_zero_not_sixty_percent(self):
        scorer=pron_score.PronScorer.__new__(pron_score.PronScorer)
        scorer.model=types.SimpleNamespace(transcribe=lambda *a,**kw:(iter([]),types.SimpleNamespace(language='fr',language_probability=1)))
        self.assertEqual(scorer.score('silent.wav','fr',None)['score'],0.)

    def test_player_absolute_ffplay_and_count_bounds(self):
        with patch.object(listening_test.subprocess,'run') as run:
            listening_test.play('/usr/bin/ffplay','clip.wav')
            self.assertIn('-nodisp',run.call_args.args[0])
        with self.assertRaises(ValueError): listening_test.sign_test_p(2,1)
        self.assertEqual(listening_test.sign_test_p(0,0),1.)


class IntegrationTests(unittest.TestCase):
    def test_final_unseen_check_uses_target_language_and_separate_text(self):
        import voice_comparator as comparator
        from probe_texts import holdout_texts, default_text
        hold = holdout_texts('en')[0]
        result = comparator.unseen_sentence('EN', default_text('en'), hold)
        self.assertIn(result, holdout_texts('en'))
        self.assertNotEqual(result, hold)
        with self.assertRaises(ValueError): comparator.unseen_sentence('ja', 'custom')
        with self.assertRaises(ValueError): comparator.unseen_sentence('en','fit','hold','fit')

    def test_audio_parser_preserves_decimal_scientific_and_chinese(self):
        import voice_comparator as comparator
        result=comparator.parse_audio_block('[1, ZH-CN, .9, 1e-1, -2.5]')
        self.assertEqual(result,dict(lang='ZH-CN',speed=.9,vol=.1,eq_low=-2.5))
        self.assertTrue(comparator.clean_text(None,'FR'))
        for block in ('[1, FR, 0]', '[1, FR, nan]', '[1, FR, garbage]', '[0, FR, 1]'):
            with self.assertRaises(ValueError): comparator.parse_audio_block(block)

    def test_unsupported_default_never_substitutes_english(self):
        import probe_texts
        with self.assertRaises(ValueError): probe_texts.default_text('ja')
        with self.assertRaises(ValueError): probe_texts.holdout_texts('ja')

    def test_missing_audio_chain_is_not_silently_scored(self):
        import voice_comparator as comparator
        fake_audio=types.SimpleNamespace(from_wav=lambda path:object())
        with patch.dict(sys.modules,{'pydub':types.SimpleNamespace(AudioSegment=fake_audio)}), \
             patch.object(comparator,'_generator_chain',return_value=None):
            with self.assertRaises(RuntimeError): comparator.apply_post('in.wav','out.wav',{}, {},'fr','bad.py')

    def test_empty_music_stream_fails_instead_of_hanging(self):
        with self.assertRaises(ValueError): brain._MusicStream(np.empty((0,2)),.2,8000)

    def test_diagnostic_cli_preserves_full_block_and_reports_suffix(self):
        generated,conditioned=[],[]
        fake=types.SimpleNamespace(load_xtts=lambda device:(None,object()),
            compute_latents=lambda *args,**kwargs:conditioned.append(kwargs) or ('gpt','speaker'),
            generate=lambda *args,**kwargs:generated.append(kwargs))
        said=truncation.LONG_TEXTS['fr'][0].split()
        asr=types.SimpleNamespace(transcribe=lambda *a,**kw:([types.SimpleNamespace(text=' '.join(said[:-5]))],None))
        modules={'xtts_clone':fake,'soundfile':types.SimpleNamespace(info=lambda path:types.SimpleNamespace(duration=2.)),
                 'faster_whisper':types.SimpleNamespace(WhisperModel=lambda *a,**kw:asr)}
        block=settings.format_block(dict(seed=0,num_beams=3,sound_norm_refs=1,max_ref_len=60,gpt_cond_len=45))
        with tempfile.TemporaryDirectory() as tmp:
            ref=Path(tmp)/'ref.wav';ref.touch()
            with patch.dict(sys.modules,modules),patch.object(sys,'argv',['test_truncation.py',str(ref),'FR',
                     '--xtts-block',block,'--temps','.7','--keep',tmp]),contextlib.redirect_stdout(io.StringIO()):
                truncation.main()
            report=json.loads((Path(tmp)/'truncation_report.json').read_text())
        self.assertEqual(conditioned,[dict(gpt_cond_len=45,gpt_cond_chunk_len=4,max_ref_len=60,sound_norm_refs=1)])
        self.assertEqual(generated[0]['seed'],0)
        self.assertEqual(generated[0]['num_beams'],3)
        self.assertFalse(generated[0]['do_sample'])
        self.assertEqual(generated[0]['temperature'],.7)
        self.assertTrue(report['samples'][0]['suspect'])
        self.assertEqual(report['samples'][0]['missing_suffix'],5)

    def test_cap_cli_uses_language_sentences_and_real_baseline(self):
        calls,gens=[],[]
        def compute(*a,**kw):
            calls.append(kw['max_ref_len'])
            return object(),kw['max_ref_len']
        def generate(model,text,lang,lat,path,**kw):
            gens.append((text,lat))
        fake=types.SimpleNamespace(load_xtts=lambda d:(None,None),compute_latents=compute,generate=generate)
        class Encoder:
            def __init__(self,**kw): pass
            def embed(self,path): return np.array([1.,0.])
            def cosine(self,a,b): return .8
        modules={'xtts_clone':fake,'speaker_identity':types.SimpleNamespace(SpeakerEncoder=Encoder),
                 'soundfile':types.SimpleNamespace(info=lambda path:types.SimpleNamespace(duration=22.))}
        with tempfile.TemporaryDirectory() as tmp:
            ref=Path(tmp)/'ref.wav';ref.touch()
            with patch.dict(sys.modules,modules),patch.object(sys,'argv',['test_max_ref_len.py',str(ref),'EN',
                 '--values','10','20','30','45','--keep',tmp]),contextlib.redirect_stdout(io.StringIO()) as output:
                cap_test.main()
            report=json.loads((Path(tmp)/'max_ref_len_report.json').read_text())
        self.assertEqual(calls,[45,10,20,30])
        self.assertTrue(gens[0][0].startswith('Hello'))
        self.assertTrue(all(lat[0] is gens[0][1][0] for _,lat in gens))
        self.assertEqual([r['max_ref_len'] for r in report['results']],[10,20,30])
        self.assertIn('effective 22.00s',output.getvalue())
        self.assertNotIn('no gain beyond the noise',output.getvalue())

    def test_diagnostic_audio_block_runs_effects_speed_and_gain(self):
        # The REAL render_voice_chain of the generator, with its stages
        # recorded: effects (limiter excluded), speed, then gain + limiter.
        from test_clonage_regressions import extract_functions
        events=[]
        class Audio:
            @staticmethod
            def from_wav(path): return Audio()
            def export(self,path,**kw): events.append(('export',path))
        chain=extract_functions('guided_meditation_generator_v23.py',{'render_voice_chain'},dict(
            process_audio=lambda audio,cfg,xp,limiter=True:events.append(('effects',xp['seed'],limiter)) or audio,
            apply_speed_rubberband=lambda audio,speed:events.append(('speed',speed)) or audio,
            apply_gain_and_limit=lambda audio,gain,limiter=True:events.append(('gain',gain,limiter)) or audio,
            print=lambda *a,**kw:None))['render_voice_chain']
        renderer=types.SimpleNamespace(DEFAULT_XTTS_PARAMS={},
            parse_voice_config=lambda *a,**kw:(1,dict(speed=.8,volume=2.5,limiter=1)),
            render_voice_chain=chain)
        with patch.dict(sys.modules,{'pydub':types.SimpleNamespace(AudioSegment=Audio),
                                    'guided_meditation_generator_v23':renderer}):
            truncation.apply_audio_block('in.wav','out.wav','[1, FR, .8, 2.5]',settings.validate(dict(seed=0)))
        self.assertEqual(events,[('effects',0,False),('speed',.8),('gain',2.5,True),('export','out.wav')])

    def test_mfa_cache_invalidates_on_model_or_command_change(self):
        from test_clonage_regressions import extract_functions
        import shutil
        calls=[]
        def run_mfa(command,wav,lines,work,model):
            calls.append((command,model))
            target=Path(work)/'result.TextGrid';target.write_text('alignment');return str(target)
        namespace=dict(os=os,tempfile=tempfile,shutil=shutil,run_mfa=run_mfa,
                       script_lines=lambda p:['bonjour le monde'],tokens=lambda s:s.split())
        function=extract_functions('vowel_pitch.py',{'cached_alignment'},namespace)['cached_alignment']
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            audio=Path(tmp)/'ref.wav';audio.touch()
            function(str(audio),'script','mfa','french_mfa')
            function(str(audio),'script','mfa','french_mfa')
            function(str(audio),'script','mfa','english_mfa')
            function(str(audio),'script','conda run -n mfa mfa','english_mfa')
        self.assertEqual(len(calls),3)

    def test_speed_one_skips_tool_and_failure_cleans_scratch(self):
        from test_clonage_regressions import extract_functions
        class Audio:
            def export(self,path,**kwargs): Path(path).write_bytes(b'audio')
        used=[]
        def fail(command,**kwargs):
            used.extend(command[-2:]);raise subprocess.CalledProcessError(1,command)
        namespace=dict(os=os,subprocess=types.SimpleNamespace(run=fail),AudioSegment=None)
        function=extract_functions('guided_meditation_generator_v23.py',{'apply_speed_rubberband'},namespace)['apply_speed_rubberband']
        audio=Audio()
        self.assertIs(function(audio,1),audio)
        with self.assertRaises(ValueError): function(audio,0)
        with self.assertRaises(subprocess.CalledProcessError): function(audio,.8)
        self.assertTrue(used)
        self.assertTrue(all(not os.path.exists(path) for path in used))

    def test_rvc_expanded_output_is_checked_and_parent_created(self):
        import rvc_convert
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);applio=root/'Applio';(applio/'.venv/bin').mkdir(parents=True)
            (applio/'core.py').touch();(applio/'.venv/bin/python').touch()
            for name in ('source.wav','model.pth','voice.index'): (root/name).touch()
            output=root/'nested/output.wav';commands=[]
            class Process:
                returncode=0
                stdout=[]
                def __init__(self,cmd,**kw):
                    commands.append(cmd);Path(cmd[cmd.index('--output_path')+1]).touch()
                def wait(self): return 0
            original=os.path.expanduser
            def expand(path): return str(output) if path=='~/new_output.wav' else original(path)
            argv=['rvc_convert.py',str(root/'source.wav'),'-o','~/new_output.wav','--model',str(root/'model.pth'),
                  '--index',str(root/'voice.index'),'--applio-dir',str(applio)]
            with patch.object(sys,'argv',argv),patch.object(rvc_convert.subprocess,'Popen',Process), \
                 patch.object(rvc_convert.os.path,'expanduser',side_effect=expand),contextlib.redirect_stdout(io.StringIO()):
                rvc_convert.main()
            self.assertTrue(output.exists())
            self.assertEqual(commands[0][commands[0].index('--output_path')+1],str(output))


if __name__=='__main__':
    unittest.main()
