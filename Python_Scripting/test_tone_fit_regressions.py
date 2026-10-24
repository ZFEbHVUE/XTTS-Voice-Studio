"""Regressions derived from the supplied log, with real DSP and model stubs."""
import contextlib
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import ltas_match as ltas
import voice_comparator as comparator
import xtts_settings as settings


class ExcerptTests(unittest.TestCase):
    def test_word_selection_returns_matching_audio_interval(self):
        words=[types.SimpleNamespace(word='Bonjour',start=3.,end=3.5),
               types.SimpleNamespace(word='Elo.',start=3.5,end=4.),
               types.SimpleNamespace(word='Encore',start=4.5,end=5.)]
        text,begin,end=comparator.transcript_excerpt([types.SimpleNamespace(words=words)],limit=12)
        self.assertEqual(text,'Bonjour Elo.')
        self.assertEqual((begin,end),(3.,4.))

    def test_empty_transcription_does_not_fabricate_a_default_sentence(self):
        with self.assertRaises(ValueError): comparator.transcript_excerpt([])

    def test_recovered_word_problem_logs_token_and_original_timestamps(self):
        segment=types.SimpleNamespace(text='Bonjour.',start=1.,end=2.,
            words=[types.SimpleNamespace(word='Bonjour.',start=2.,end=1.)])
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(comparator.transcript_excerpt([segment]),('Bonjour.',1.,2.))
        for detail in ('segment 1',"text='Bonjour.'",'start=2.0','end=1.0','complete segment'):
            self.assertIn(detail,output.getvalue())

    def test_unrecoverable_timestamps_are_logged_before_error(self):
        segment=types.SimpleNamespace(text='Bonjour.',start=None,end=None,
            words=[types.SimpleNamespace(word='Bonjour.',start=float('nan'),end=1.)])
        output=io.StringIO()
        with contextlib.redirect_stdout(output),self.assertRaisesRegex(ValueError,'No usable timestamped excerpt'):
            comparator.transcript_excerpt([segment])
        self.assertIn('start=nan, end=1.0',output.getvalue())
        self.assertIn('complete segment unavailable',output.getvalue())

    def test_timestamp_diagnostics_are_bounded_for_many_zero_duration_words(self):
        words=[types.SimpleNamespace(word='a',start=1.,end=1.) for _ in range(100)]
        words.append(types.SimpleNamespace(word='Bonjour.',start=1.,end=2.))
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            text,start,end=comparator.transcript_excerpt([types.SimpleNamespace(words=words)])
        self.assertEqual((start,end),(1.,2.))
        self.assertEqual(text.count('a '),100)
        self.assertEqual(output.getvalue().count('[INFO] ASR segment'),8)
        self.assertIn('92 further timestamp issue(s)',output.getvalue())

    def test_long_segment_without_words_is_not_partially_paired(self):
        segment=types.SimpleNamespace(text='Long '*100,start=0.,end=30.)
        with self.assertRaises(ValueError): comparator.transcript_excerpt([segment])

    def test_zero_duration_words_are_grouped_without_losing_text(self):
        words=[types.SimpleNamespace(word='Et',start=2.,end=2.),
               types.SimpleNamespace(word='bonjour',start=2.,end=2.5),
               types.SimpleNamespace(word='!',start=2.5,end=2.5),
               types.SimpleNamespace(word='Encore.',start=3.,end=4.)]
        with contextlib.redirect_stdout(io.StringIO()):
            text,start,end=comparator.transcript_excerpt([types.SimpleNamespace(words=words)],limit=12)
        self.assertEqual((text,start,end),('Et bonjour !',2.,2.5))

    def test_invalid_word_alignment_falls_back_to_complete_timed_segment(self):
        for begin,end in ((None,1.),(float('nan'),1.),(-1.,1.),(2.,1.)):
            with self.subTest(begin=begin,end=end):
                words=[types.SimpleNamespace(word='Bonjour',start=begin,end=end)]
                segment=types.SimpleNamespace(text='Bonjour à vous.',start=2.,end=4.,words=words)
                with contextlib.redirect_stdout(io.StringIO()):
                    result=comparator.transcript_excerpt([segment])
                self.assertEqual(result,('Bonjour à vous.',2.,4.))

    def test_only_zero_duration_words_use_segment_interval(self):
        segment=types.SimpleNamespace(text='Bonjour.',start=1.,end=2.,
            words=[types.SimpleNamespace(word='Bonjour.',start=1.,end=1.)])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(comparator.transcript_excerpt([segment]),('Bonjour.',1.,2.))

    def test_overlong_broken_first_segment_can_be_skipped_for_a_valid_excerpt(self):
        first=types.SimpleNamespace(text='Long '*100,start=0.,end=30.,
            words=[types.SimpleNamespace(word='Long',start=None,end=None)])
        second=types.SimpleNamespace(text='Bonjour.',start=31.,end=33.)
        with contextlib.redirect_stdout(io.StringIO()):
            result=comparator.transcript_excerpt([first,second])
        self.assertEqual(result,('Bonjour.',31.,33.))

    def test_broken_middle_segment_does_not_create_an_untranscribed_audio_gap(self):
        first=types.SimpleNamespace(text='Bonjour.',start=0.,end=1.)
        broken=types.SimpleNamespace(text='Manquant.',start=None,end=None)
        last=types.SimpleNamespace(text='Encore.',start=3.,end=4.)
        self.assertEqual(comparator.transcript_excerpt([first,broken,last]),('Bonjour.',0.,1.))

    def test_zero_duration_group_is_not_partially_selected_at_character_limit(self):
        first=types.SimpleNamespace(words=[types.SimpleNamespace(word='Un',start=0.,end=0.),
            types.SimpleNamespace(word='longmot',start=0.,end=1.)])
        second=types.SimpleNamespace(text='Oui.',start=2.,end=3.)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(comparator.transcript_excerpt([first,second],limit=5),('Oui.',2.,3.))

    def test_overlapping_word_intervals_keep_full_audio_coverage(self):
        words=[types.SimpleNamespace(word='Bonjour',start=0.,end=2.),
            types.SimpleNamespace(word='vous',start=1.,end=1.5)]
        self.assertEqual(comparator.transcript_excerpt([types.SimpleNamespace(words=words)]),
                         ('Bonjour vous',0.,2.))


class ToneTests(unittest.TestCase):
    def noise(self): return np.random.default_rng(4).normal(0,.01,48000)

    def test_response_uses_clone_rate_and_cumulative_gains_are_bounded(self):
        reference=self.noise();seen=[]
        def render(config): seen.append(dict(config));return reference.copy()
        def proposed(*a,**kw):
            self.assertEqual(kw['fs'],24000)
            return dict(eq_low=4,eq_mid=4,eq_high=4,hp=0,lp=0)
        with patch.object(ltas,'fit_eq_ls',side_effect=proposed),contextlib.redirect_stdout(io.StringIO()):
            result,history=comparator.fit_rendered_tone(reference,44100,24000,
                dict(eq_low=3,eq_mid=3,eq_high=3,hp=0,lp=0,vol=0),render,eq_limit=4)
        self.assertTrue(all(abs(config[k])<=4 for config in seen for k in ('eq_low','eq_mid','eq_high')))
        # Predicted improvement is not enough: unchanged real spectra reject it.
        self.assertEqual(result['eq_high'],3)
        self.assertFalse(history[-1]['accepted'])

    def test_worse_actual_render_is_rejected_even_if_solver_predicts_improvement(self):
        original=self.noise()
        altered=original+np.sin(np.arange(len(original))*.05)*.02
        def render(config): return altered if config.get('eq_high') else original
        proposal=dict(eq_low=0,eq_mid=0,eq_high=4,hp=0,lp=0)
        with patch.object(ltas,'fit_eq_ls',return_value=proposal),contextlib.redirect_stdout(io.StringIO()):
            result,history=comparator.fit_rendered_tone(original,24000,24000,
                dict(eq_low=0,eq_mid=0,eq_high=0,hp=0,lp=0,vol=0),render)
        self.assertEqual(result['eq_high'],0)
        self.assertGreater(history[-1]['after'],history[-1]['before'])

    @unittest.skipUnless(shutil.which('ffmpeg'),'ffmpeg unavailable')
    def test_actual_ffmpeg_bell_fit_is_improved(self):
        raw=self.noise().astype('<f4')
        def eq(audio,gain):
            result=subprocess.run(['ffmpeg','-v','error','-f','f32le','-ar','24000','-ac','1',
                '-i','pipe:0','-af',f'equalizer=f=5000:t=h:w=3000:g={gain}',
                '-f','f32le','pipe:1'],input=audio.tobytes(),capture_output=True,check=True)
            return np.frombuffer(result.stdout,dtype='<f4').copy()
        reference=eq(raw,3)
        def render(config): return eq(raw,config.get('eq_high',0))*10**(config.get('vol',0)/20)
        with contextlib.redirect_stdout(io.StringIO()):
            result,history=comparator.fit_rendered_tone(reference,24000,24000,
                dict(eq_low=0,eq_mid=0,eq_high=0,hp=0,lp=0,vol=0),render)
        self.assertTrue(history[0]['accepted'])
        self.assertLess(history[-1]['after'],history[0]['before'])
        self.assertAlmostEqual(result['eq_high'],3,delta=.3)


class IntegrationTests(unittest.TestCase):
    def test_joint_search_restores_requested_lengths_after_short_cap_wins(self):
        from test_clonage_regressions import OptimizerTests
        def score(record):
            cond=record['conditioning']
            combo=(cond['max_ref_len'],cond['gpt_cond_len'],cond['gpt_cond_chunk_len'])
            if combo==(45,30,8): return .99
            if cond['max_ref_len']==20: return .7
            return .5
        result,log,conditions,_=OptimizerTests().run_optimizer([
            '--method','coord','--rounds','0','--gpt-cond-lens','20 30',
            '--gpt-chunks','4 8','--max-ref-lens','20 30 45','--cond-budget','48',
            '--budget','2','--no-holdout'],score=score)
        self.assertEqual((result['params']['max_ref_len'],result['params']['gpt_cond_len'],
                          result['params']['gpt_cond_chunk_len']),(45,30,8))

    def test_joint_search_revisits_reference_cap_with_new_gpt_chunk(self):
        from test_clonage_regressions import OptimizerTests
        def score(record):
            cond=record['conditioning']
            return .99 if (cond['max_ref_len'],cond['gpt_cond_len'],cond['gpt_cond_chunk_len'])==(45,20,8) else .5
        result,log,conditions,_=OptimizerTests().run_optimizer([
            '--method','coord','--rounds','0','--gpt-cond-lens','20 30',
            '--gpt-chunks','4 8','--max-ref-lens','30 45','--cond-budget','40',
            '--budget','2','--no-holdout'],score=score)
        self.assertEqual((result['params']['max_ref_len'],result['params']['gpt_cond_len'],
                          result['params']['gpt_cond_chunk_len']),(45,20,8))
        self.assertIn('C5',log)
        self.assertLessEqual(result['generations']['conditioning'],40)

    def test_full_comparator_pairs_reference_uses_render_rate_and_exports_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);reference=root/'reference.wav'
            raw=np.sin(2*np.pi*200*np.arange(48000)/24000).astype(np.float32)*.02
            source=resample_poly(raw,147,80).astype(np.float32)
            wavfile.write(reference,44100,source)
            raw_path,base_path,fit_path=(root/name for name in ('raw.wav','base.wav','fit.wav'))
            def info(path):
                rate,data=wavfile.read(path)
                return types.SimpleNamespace(samplerate=rate,duration=len(data)/rate)
            def read(path,**kwargs):
                rate,data=wavfile.read(path)
                return data,rate
            clone=types.SimpleNamespace(load_xtts=lambda device:(None,None),
                compute_latents=lambda *a,**kw:(None,None),
                generate=lambda model,text,lang,lat,path,**kw:wavfile.write(path,24000,raw))
            # The previous release aborted the whole CLI on this zero-length token.
            words=[types.SimpleNamespace(word='Eh',start=.5,end=.5),
                types.SimpleNamespace(word='Bonjour',start=.5,end=1.),types.SimpleNamespace(word='Elo.',start=1.,end=1.5)]
            asr=types.SimpleNamespace(transcribe=lambda *a,**kw:([types.SimpleNamespace(words=words)],None))
            modules={'xtts_clone':clone,'torch':types.SimpleNamespace(),
                'soundfile':types.SimpleNamespace(info=info,read=read),
                'faster_whisper':types.SimpleNamespace(WhisperModel=lambda *a,**kw:asr)}
            seen=[]
            real_fit=comparator.fit_rendered_tone
            def fit(ref,ref_sr,render_sr,*a,**kw):
                seen.append((len(ref),ref_sr,render_sr))
                return real_fit(ref,ref_sr,render_sr,*a,**kw)
            argv=['voice_comparator.py',str(reference),'FR','--xtts-block',settings.format_block({'seed':0}),
                '--audio-block','[1, FR, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]',
                '--auto-text','--device','cpu','--no-fit-dynamics','--no-identity-check',
                '--output',str(base_path),'--output-optimised',str(fit_path),'--output-raw',str(raw_path)]
            with patch.dict(sys.modules,modules),patch.object(sys,'argv',argv), \
                 patch.object(comparator,'apply_post',side_effect=lambda src,dst,*a:shutil.copyfile(src,dst)), \
                 patch.object(comparator,'fit_rendered_tone',side_effect=fit), \
                 contextlib.redirect_stdout(io.StringIO()):
                comparator.main()
            self.assertEqual(seen,[(44100,44100,24000)])
            self.assertTrue(all(path.exists() for path in (raw_path,base_path,fit_path)))
            rate,data=wavfile.read(raw_path)
            self.assertEqual(rate,24000)
            np.testing.assert_array_equal(data,raw)


    def test_comparator_refits_a_given_clone_without_xtts(self):
        # A second tone fit (another EQ weighting) must work on the very same
        # clone and the same stretch of the reference: --raw-clone and
        # --reference-interval, and no model to load when nothing generates.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);reference=root/'reference.wav'
            tone=np.sin(2*np.pi*200*np.arange(48000)/24000).astype(np.float32)*.02
            wavfile.write(reference,44100,resample_poly(tone,147,80).astype(np.float32))
            given=root/'given_raw.wav'
            wavfile.write(given,24000,tone)
            raw_path,base_path,fit_path=(root/name for name in ('raw.wav','base.wav','fit.wav'))
            def info(path):
                rate,data=wavfile.read(path)
                return types.SimpleNamespace(samplerate=rate,duration=len(data)/rate)
            def read(path,**kwargs):
                rate,data=wavfile.read(path)
                return data,rate
            def forbidden(*a,**kw):
                raise AssertionError('XTTS must not run on a reused clone')
            clone=types.SimpleNamespace(load_xtts=forbidden,compute_latents=forbidden,generate=forbidden)
            modules={'xtts_clone':clone,'torch':types.SimpleNamespace(),
                'soundfile':types.SimpleNamespace(info=info,read=read)}
            seen=[]
            real_fit=comparator.fit_rendered_tone
            def fit(ref,ref_sr,render_sr,*a,**kw):
                seen.append((len(ref),ref_sr,render_sr,kw.get('weighting')))
                return real_fit(ref,ref_sr,render_sr,*a,**kw)
            argv=['voice_comparator.py',str(reference),'FR','--xtts-block',settings.format_block({'seed':0}),
                '--audio-block','[1, FR, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]',
                '--raw-clone',str(given),'--text','Bonjour Elo.','--reference-interval','0.5','1.5',
                '--eq-weighting','blend','--device','cpu','--no-fit-dynamics','--no-identity-check',
                '--output',str(base_path),'--output-optimised',str(fit_path),'--output-raw',str(raw_path)]
            out=io.StringIO()
            with patch.dict(sys.modules,modules),patch.object(sys,'argv',argv), \
                 patch.object(comparator,'apply_post',side_effect=lambda src,dst,*a:shutil.copyfile(src,dst)), \
                 patch.object(comparator,'fit_rendered_tone',side_effect=fit), \
                 contextlib.redirect_stdout(out):
                comparator.main()
            # One second of the 44.1 kHz reference: the given stretch, not all of it.
            self.assertEqual(seen,[(44100,44100,24000,'blend')])
            self.assertIn('[*] Fit input: {"text": "Bonjour Elo.", "interval": [0.5, 1.5]}',out.getvalue())
            self.assertIn('Clone reused from given_raw.wav (XTTS not loaded)',out.getvalue())
            np.testing.assert_array_equal(wavfile.read(raw_path)[1],tone)

    def test_reused_clone_arguments_are_checked_before_anything_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference=Path(tmp)/'reference.wav'
            wavfile.write(reference,16000,np.zeros(16000,np.float32))
            base=['voice_comparator.py',str(reference),'FR','--xtts-block',settings.format_block({}),
                  '--audio-block','[1, FR, 1]']
            for extra in (['--raw-clone',str(Path(tmp)/'missing.wav')],
                          ['--auto-text','--reference-interval','0','1'],
                          ['--text','Bonjour.','--reference-interval','2','1']):
                with self.subTest(extra=extra),patch.object(sys,'argv',base+extra), \
                     contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as cm:
                    comparator.main()
                self.assertEqual(cm.exception.code,2)


if __name__=='__main__': unittest.main()
