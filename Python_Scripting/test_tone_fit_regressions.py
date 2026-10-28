"""Regressions derived from the supplied log, with real DSP and model stubs."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, resample_poly, sosfilt
import ltas_match as ltas
from audio_parameter_search import AUDIO_KEYS as AUDIO_FIELDS
import voice_comparator as comparator
import xtts_settings as settings
from test_render_search_regressions import AudioSegment, CHAIN, generator_namespace, needs_audio


def shaped_noise(sr, seconds, seed, tilt_db=0.):
    """Noise with a speech-like long-term spectrum (rising to 150 Hz, -6 dB per
    octave above 600 Hz, a sibilance bump at 6 kHz), tilted by tilt_db from
    3 to 7 kHz."""
    n=int(sr*seconds)
    spectrum=np.fft.rfft(np.random.default_rng(seed).standard_normal(n))
    f=np.maximum(np.fft.rfftfreq(n,1/sr),1.)
    db=(-10*np.log10(1+(150/f)**4)-10*np.log10(1+(f/600)**2)
        +5*np.exp(-.5*(np.log2(f/6000)/.35)**2)
        +tilt_db*np.clip(np.log2(f/3000)/np.log2(7000/3000),0,1))
    y=np.fft.irfft(spectrum*10**(db/20),n)
    return .05*y/np.std(y)


def exact_chain(raw, config, fs=24000, artifact=None, ratio=(147, 80)):
    """hp, lp, EQ and volume as the generator applies them (pydub's RC
    recursions, FFmpeg's RBJ bells), resampled to the reference's rate
    (ratio: 44.1 kHz by default, (2, 3) for 16 kHz). `artifact` is added when
    the low-pass is off: something it hid, which no model of the chain knows
    about."""
    y=np.asarray(raw,float)
    if config.get('hp'):
        a=1/(1+2*np.pi*config['hp']/fs); y=lfilter([a,-a],[1,-a],y)
    if config.get('lp'):
        a=1/(1+fs/(2*np.pi*config['lp'])); y=lfilter([a],[1,a-1],y)
    for key,f0,bw in ltas.EQ_BANDS:
        if config.get(key):
            A=10**(config[key]/40); w0=2*np.pi*f0/fs; alpha=np.sin(w0)*bw/(2*f0); c=np.cos(w0)
            b=np.array([1+alpha*A,-2*c,1-alpha*A]); den=np.array([1+alpha/A,-2*c,1-alpha/A])
            y=lfilter(b/den[0],den/den[0],y)
    if artifact is not None and not config.get('lp'):
        y=y+artifact
    return resample_poly(y*10**(config.get('vol',0)/20),*ratio)


def harmonic_voice(sr, seconds, f0, phase, h1_db=0., noise_db=-45., seed=0, top=7000.):
    """A voiced source under four formants: the harmonics of a gliding F0
    (amplitude 1/k) up to `top` Hz at any rate, syllable gates, a little
    noise. h1_db raises the fundamental alone, the balance a clone can get
    wrong."""
    rng=np.random.default_rng(seed)
    t=np.arange(int(sr*seconds))/sr
    pitch=f0*(1+.08*np.sin(2*np.pi*.31*t+phase)+.04*np.sin(2*np.pi*1.13*t+2*phase))
    angle=2*np.pi*np.cumsum(pitch)/sr
    def formants(f):
        f=np.maximum(f,1.)
        return sum(g*np.exp(-.5*(np.log2(f/fc)/bw)**2) for fc,g,bw in
                   ((800,12,.35),(1900,9,.3),(2800,6,.25),(3600,3,.25)))
    y=np.zeros_like(t)
    top=min(top,sr/2-200)
    for k in range(1,int(top//(f0*.85))+1):
        fk=k*pitch
        amp=10**((formants(fk)+(h1_db if k==1 else 0))/20)/k
        y+=np.where(fk<top,amp,0)*np.sin(k*angle+rng.uniform(0,2*np.pi))
    gate=np.convolve((np.sin(2*np.pi*2.7*t+phase)>-.4).astype(float),np.ones(int(.03*sr))/int(.03*sr),'same')
    y=y*gate/np.std(y*gate)
    noise=lfilter([1],[1,-.95],rng.standard_normal(len(t)))
    y=y+10**(noise_db/20)*noise/np.std(noise)
    return .05*y/np.std(y)


def low_share(y, sr):
    """Share of the energy below 300 Hz, of the energy below 8 kHz."""
    power=np.abs(np.fft.rfft(y))**2;f=np.fft.rfftfreq(len(y),1/sr)
    return float(power[f<300].sum()/power[f<8000].sum())


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
        def proposed(grid,*a,**kw):
            self.assertEqual(kw['fs'],24000)
            # Beyond the limit on purpose: the comparator still clips.
            return dict(eq_low=7,eq_mid=7,eq_high=7),np.zeros(len(grid))
        with patch.object(ltas,'fit_eq_psd',side_effect=proposed),contextlib.redirect_stdout(io.StringIO()):
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
        proposal=dict(eq_low=0,eq_mid=0,eq_high=4),np.zeros(ltas.N_GRID)
        with patch.object(ltas,'fit_eq_psd',return_value=proposal),contextlib.redirect_stdout(io.StringIO()):
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
        # max_ref 20 wins C0 and drops gcl 30 from C1 (one 20 s file cannot
        # give 30 s of conditioning); chunk 8 then wins C2. C5 rebuilds its
        # grid from the requested lengths, so gcl 30 comes back with the
        # longer cap, within C5's nearest new renders.
        from test_clonage_regressions import OptimizerTests
        def score(record):
            cond=record['conditioning']
            combo=(cond['max_ref_len'],cond['gpt_cond_len'],cond['gpt_cond_chunk_len'])
            if combo==(45,30,8): return .99
            return (.7 if cond['max_ref_len']==20 else .5)+(.05 if cond['gpt_cond_chunk_len']==8 else 0)
        result,log,conditions,_=OptimizerTests().run_optimizer([
            '--method','coord','--rounds','0','--gpt-cond-lens','20 30',
            '--gpt-chunks','4 8','--max-ref-lens','20 45','--cond-budget','48',
            '--budget','2','--no-holdout'],score=score)
        self.assertIn('-> C0 max_ref_len: ADOPTED gcl=20 chunk=4 norm=0 refs=1 max_ref=20',log)
        self.assertIn('-> C5 joint checks: ADOPTED gcl=30 chunk=8 norm=0 refs=1 max_ref=45',log)
        self.assertEqual((result['params']['max_ref_len'],result['params']['gpt_cond_len'],
                          result['params']['gpt_cond_chunk_len']),(45,30,8))

    def test_joint_search_revisits_reference_cap_with_new_gpt_chunk(self):
        # C0 tried max_ref 45 under chunk 4, where it does nothing; C1 and C2
        # then move to gcl 20 and chunk 8, under which 45 is the best cap.
        from test_clonage_regressions import OptimizerTests
        def score(record):
            cond=record['conditioning']
            if (cond['max_ref_len'],cond['gpt_cond_len'],cond['gpt_cond_chunk_len'])==(45,20,8): return .99
            return .5+(.05 if cond['gpt_cond_len']==20 else 0)+(.1 if cond['gpt_cond_chunk_len']==8 else 0)
        result,log,conditions,_=OptimizerTests().run_optimizer([
            '--method','coord','--rounds','0','--gpt-cond-lens','20 30',
            '--gpt-chunks','4 8','--max-ref-lens','30 45','--cond-budget','40',
            '--budget','2','--no-holdout'],score=score)
        self.assertIn('-> C2 gpt_cond_chunk_len: ADOPTED gcl=20 chunk=8',log)
        self.assertIn('-> C5 joint checks: ADOPTED gcl=20 chunk=8 norm=0 refs=1 max_ref=45',log)
        self.assertEqual((result['params']['max_ref_len'],result['params']['gpt_cond_len'],
                          result['params']['gpt_cond_chunk_len']),(45,20,8))
        self.assertLessEqual(result['generations']['conditioning'],40)

    def test_joint_search_screens_only_the_nearest_combinations(self):
        # A pure three-way interaction, which no single setting hints at, three
        # changes away from the incumbent: C5 stops after its 8 new renders
        # (the price of keeping its budget for the confirmation) and says how
        # far it got.
        from test_clonage_regressions import OptimizerTests
        def score(record):
            cond=record['conditioning']
            return .99 if (cond['max_ref_len'],cond['gpt_cond_len'],cond['gpt_cond_chunk_len'])==(45,20,8) else .5
        result,log,conditions,_=OptimizerTests().run_optimizer([
            '--method','coord','--rounds','0','--gpt-cond-lens','20 30',
            '--gpt-chunks','4 8','--max-ref-lens','30 45','--cond-budget','40',
            '--budget','2','--no-holdout'],score=score)
        self.assertIn('C5 screened the nearest 14 of 16 combinations (8 new render(s), cap 8)',log)
        self.assertEqual((result['params']['max_ref_len'],result['params']['gpt_cond_len'],
                          result['params']['gpt_cond_chunk_len']),(30,30,4))
        self.assertEqual(result['truncated_stages'],{})    # a cap by design, not a cut

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


class LowpassFitTests(unittest.TestCase):
    """3.7.1: pass 1 weighs the low-pass off against the block's own filters
    with their exact responses, and the render decides."""
    START=dict(vol=0,eq_low=0,eq_mid=0,eq_high=0,hp=0,lp=9000)

    @classmethod
    def setUpClass(cls):
        cls.reference=shaped_noise(44100,4,1)
        cls.same=shaped_noise(24000,4,2)

    def fit(self,raw,start,reference=None,artifact=None,render=None):
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            result,history=comparator.fit_rendered_tone(
                self.reference if reference is None else reference,44100,24000,start,
                render or (lambda config:exact_chain(raw,config,artifact=artifact)))
        return result,history,output.getvalue()

    def test_a_lowpass_that_only_dulls_the_clone_is_switched_off(self):
        result,history,log=self.fit(self.same,self.START)
        self.assertEqual(result['lp'],0)
        # The EQ was fitted on the curve without the filter: no boost to undo it.
        self.assertLessEqual(abs(result['eq_high']),.5)
        first=history[0]
        self.assertTrue(first['accepted'])
        self.assertLess(first['after'],first['before']-.5)
        self.assertAlmostEqual(first['predicted'],first['after'],delta=.05)
        self.assertIn('low-pass 9000 Hz -> off',log)

    def test_a_block_pasted_back_gives_its_boost_back(self):
        start=dict(self.START,eq_high=4)
        grid,ref_db=ltas.compute_ltas(self.reference,44100,fmax=8000)
        _,clone_db=ltas.compute_ltas(exact_chain(self.same,start),44100,fmax=8000)
        # The bell hides part of the filter's cut: above 6 kHz the clone lacks
        # less than the 3 dB that makes derive_hp_lp switch the low-pass off ...
        self.assertEqual(ltas.derive_hp_lp(grid,ref_db,clone_db,0,9000),(0,9000))
        result,history,log=self.fit(self.same,start)
        # ... the predicted error switches it off all the same.
        self.assertEqual(result['lp'],0)
        self.assertLessEqual(result['eq_high'],1)
        self.assertLess(history[0]['after'],history[0]['before']-.3)

    def test_when_the_render_disagrees_the_block_filters_are_tried(self):
        sos=butter(4,[5000,8000],btype='band',fs=24000,output='sos')
        hiss=sosfilt(sos,np.random.default_rng(7).standard_normal(len(self.same)))
        result,history,log=self.fit(self.same,self.START,artifact=.03*hiss/np.std(hiss))
        off,kept=history[0],history[1]
        self.assertEqual((off['pass_number'],off['config']['lp'],off['accepted']),(1,0,False))
        self.assertGreater(off['after'],off['predicted']+.5)     # what the model could not know
        self.assertEqual((kept['pass_number'],kept['config']['lp'],kept['accepted']),(1,9000,True))
        self.assertLess(kept['after'],kept['before'])
        self.assertEqual(result['lp'],9000)
        self.assertIn('filters unchanged; kept',log)

    def test_a_filter_change_the_render_does_not_show_is_not_kept(self):
        # Every render the same: switching the low-pass off measures nothing,
        # so the block keeps its own, like any change that does not improve.
        dull=exact_chain(self.same,dict(lp=9000))
        zero=dict(eq_low=0,eq_mid=0,eq_high=0),np.zeros(ltas.N_GRID)
        with patch.object(ltas,'fit_eq_psd',return_value=zero):
            result,history,_=self.fit(None,self.START,render=lambda config:dull)
        self.assertEqual([(h['config']['lp'],h['accepted']) for h in history],[(0,False),(9000,True)])
        self.assertEqual(result['lp'],9000)

    def test_a_bright_clone_keeps_its_lowpass(self):
        result,history,_=self.fit(shaped_noise(24000,4,2,tilt_db=4),self.START)
        self.assertEqual(result['lp'],9000)
        self.assertTrue(all(h['config']['lp']==9000 for h in history))   # off is not even rendered

    def test_a_filter_change_earns_a_second_pass(self):
        # The clone is the reference behind a low-pass: switching it off is
        # the whole fit, the EQ does not move, and pass 2 still checks it.
        reference=exact_chain(self.same,{})
        result,history,_=self.fit(self.same,self.START,reference=reference)
        self.assertEqual(result['lp'],0)
        self.assertEqual([h['pass_number'] for h in history],[1,2])
        self.assertEqual([history[0]['config'][k] for k in ('eq_low','eq_mid','eq_high')],[0,0,0])


@needs_audio
class RealChainLowpassTests(unittest.TestCase):
    """The same, through the generator's own render chain (pydub, FFmpeg)."""

    def test_lowpass_off_and_flat_treble_through_the_real_chain(self):
        chain=generator_namespace(CHAIN)['render_voice_chain']
        reference,raw=shaped_noise(44100,4,1),shaped_noise(24000,4,2)
        segment=AudioSegment((np.clip(raw,-1,1)*32767).astype(np.int16).tobytes(),
                             frame_rate=24000,sample_width=2,channels=1)
        def render(c):
            config=dict(speed=1.,volume=float(c['vol']),eq_low=float(c['eq_low']),eq_mid=float(c['eq_mid']),
                        eq_high=float(c['eq_high']),highpass=float(c['hp']),lowpass=float(c['lp']),
                        noise_reduction=0,compression=0,deesser=0,reverb=0,noise_gate=0,pan=0,
                        limiter=1,language='fr')
            out=chain(segment,config,dict(trim_start=0,trim_end=0,fade_in=0,fade_out=0))
            return resample_poly(np.array(out.get_array_of_samples(),dtype=float)/32768,147,80)
        with contextlib.redirect_stdout(io.StringIO()):
            result,history=comparator.fit_rendered_tone(reference,44100,24000,
                dict(vol=0,eq_low=0,eq_mid=0,eq_high=0,hp=90,lp=9000),render)
        self.assertEqual(result['lp'],0)
        self.assertLessEqual(abs(result['eq_high']),.5)
        self.assertLess(history[0]['after'],history[0]['before']-.5)
        self.assertAlmostEqual(history[0]['predicted'],history[0]['after'],delta=.1)


class SmoothedSpectrumTests(unittest.TestCase):
    """3.7.2: spectra compared over bands, and the EQ fitted on the PSD."""

    def test_bands_span_a_harmonic_spacing_then_a_third_of_an_octave(self):
        widths=ltas.smoothing_widths(np.array([100.,1000.,2000.,8000.]))
        np.testing.assert_allclose(widths[:2],[250.,250.])
        np.testing.assert_allclose(widths[2:],np.array([2000.,8000.])*(2**(1/6)-2**(-1/6)))

    def test_flat_stays_flat_and_floor_zero_reads_point_by_point(self):
        freqs=np.linspace(0,8000,2049);grid=np.geomspace(80,7999,64)
        np.testing.assert_allclose(ltas.grid_db(freqs,np.ones_like(freqs),grid),0,atol=1e-9)
        psd=1+np.cos(freqs/20)**2
        np.testing.assert_allclose(ltas.grid_db(freqs,psd,grid,floor_hz=0),
                                   np.interp(grid,freqs,10*np.log10(psd)))

    def test_psd_fit_finds_a_bell_and_keeps_a_band_at_its_bound(self):
        freqs=np.linspace(0,8000,2049);flat=np.ones_like(freqs);grid=np.geomspace(80,7999,256)
        bells=lambda g:10**(ltas.eq_peaking_db(freqs,24000,g)/10)
        target=ltas.grid_db(freqs,flat*bells([0,-1.5,3]),grid)
        gains,curve=ltas.fit_eq_psd(grid,target,freqs,flat,fs=24000,limit=4)
        self.assertEqual(gains,dict(eq_low=0.,eq_mid=-1.5,eq_high=3.))
        self.assertLess(ltas.spectral_shape_error(grid,target,curve),.01)
        # A band asked for -6 dB stays at its -4 dB bound, and the others are
        # fitted knowing it: better than the step fitted freely and clipped
        # afterwards (3.7.1), which gave -4/+2/0 here.
        target=ltas.grid_db(freqs,flat*bells([-6,2,0]),grid)
        rendered=flat*bells([-4,0,0])
        gains,curve=ltas.fit_eq_psd(grid,target,freqs,rendered,fs=24000,limit=4,start=[-4,0,0])
        self.assertEqual(gains['eq_low'],-4.)
        clipped=ltas.grid_db(freqs,flat*bells([-4,2,0]),grid)
        self.assertLess(ltas.spectral_shape_error(grid,target,curve),
                        ltas.spectral_shape_error(grid,target,clipped)-.05)


class HarmonicVoiceTests(unittest.TestCase):
    """3.7.2: below ~1 kHz a voice's spectrum is a comb that follows the melody."""

    @classmethod
    def setUpClass(cls):
        cls.reference=harmonic_voice(16000,6,204,0.,noise_db=-40,seed=1)
        cls.clones={h1:harmonic_voice(24000,6,208,1.3,h1_db=h1,noise_db=-55,seed=2) for h1 in (0,4)}

    def render(self,h1,config):
        return exact_chain(self.clones[h1],config,ratio=(2,3))

    def fit(self,h1):
        with contextlib.redirect_stdout(io.StringIO()):
            return comparator.fit_rendered_tone(self.reference,16000,24000,
                dict(vol=0,eq_low=0,eq_mid=0,eq_high=0,hp=0,lp=0),lambda c:self.render(h1,c))

    def test_point_by_point_the_comb_hides_a_real_difference(self):
        errors={}
        for h1 in (0,4):
            pair=[]
            for floor in (0,ltas.SMOOTH_FLOOR_HZ):
                grid,ref_db=ltas.compute_ltas(self.reference,16000,fmax=7999,floor_hz=floor)
                _,clone_db=ltas.compute_ltas(self.render(h1,{}),16000,fmax=7999,floor_hz=floor)
                pair.append(ltas.spectral_shape_error(grid,ref_db,clone_db))
            errors[h1]=pair
        # Point by point: over 3 dB, with or without 4 dB more fundamental.
        self.assertGreater(errors[0][0],3)
        self.assertAlmostEqual(errors[0][0],errors[4][0],delta=.3)
        # Smoothed: the same voice is close, the stronger fundamental is not.
        self.assertLess(errors[0][1],.5)
        self.assertGreater(errors[4][1],errors[0][1]+.5)

    def test_another_intonation_is_not_a_tone_to_correct(self):
        result,_=self.fit(0)
        self.assertTrue(all(abs(result[k])<=.5 for k in ('eq_low','eq_mid','eq_high')),result)

    def test_a_too_strong_fundamental_is_cut_not_boosted(self):
        result,history=self.fit(4)
        self.assertLessEqual(result['eq_low'],-3)
        reference=low_share(self.reference,16000)
        before,after=(low_share(self.render(4,c),16000) for c in ({},result))
        self.assertLess(abs(after-reference),abs(before-reference)/2)
        # The bells act on the PSD before the smoothing, as on the audio.
        self.assertAlmostEqual(history[0]['predicted'],history[0]['after'],delta=.02)


@unittest.skipUnless(importlib.util.find_spec('librosa') is not None,'librosa not installed')
class AnalyserFilterTests(unittest.TestCase):
    def test_no_lowpass_whatever_the_voice_type(self):
        # 3.7 wrote 7000 to 9000 Hz by voice type: a treble tilt on every clone.
        import voice_analyser as analyser
        sr=24000;t=np.arange(4*sr)/sr
        seen={}
        with tempfile.TemporaryDirectory() as tmp:
            for f0 in (95,140,180,250):
                phase=2*np.pi*np.cumsum(f0*(1+.03*np.sin(2*np.pi*5*t)))/sr
                y=sum(.6/k*np.sin(k*phase) for k in range(1,25) if k*f0<sr/2)
                syllables=np.convolve((np.sin(2*np.pi*3.5*t)>-.3).astype(float),np.ones(240)/240,mode='same')
                path=Path(tmp)/f'v{f0}.wav'
                wavfile.write(path,sr,(.2*y*syllables).astype(np.float32))
                with contextlib.redirect_stdout(io.StringIO()):
                    params,stats=analyser.analyse_voice(str(path),fast=True,use_praat=False)
                seen[stats['voice_type'].split(' /')[0]]=(params['highpass'],params['lowpass'])
        self.assertEqual(seen,{'bass':(55,0),'baritone':(65,0),'tenor':(75,0),'soprano':(90,0)})


class ComparatorWordingTests(unittest.TestCase):
    """From a review of 3.7.1."""

    def test_the_tab_runs_three_passes_and_names_its_threshold(self):
        source=(Path(__file__).resolve().parent/'xtts_studio.py').read_text(encoding='utf-8')
        self.assertIn("v_cmp_iter = tk.StringVar(value='3')",source)   # as the script and the pipeline
        self.assertIn('Stop if EQ/volume change <',source)             # what --conv-threshold is
        self.assertNotIn('Stop if score delta',source)

    def test_one_unseen_sentence_is_not_sold_as_the_whole_meditation(self):
        source=Path(comparator.__file__).read_text(encoding='utf-8')
        self.assertNotIn('your generated meditation will resemble',source)
        self.assertIn('one ECAPA score does not predict an entire meditation',source)


class ComparatorStageTests(unittest.TestCase):
    """The identity stages of main(), with the encoder and the render stubbed:
    `identity` scores the block rendered last."""

    def run_main(self,block,extra,identity=lambda block:.8, return_report=False, post_error=None,
                 presets=None, dynamics=False):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);reference=root/'reference.wav'
            tone=np.sin(2*np.pi*200*np.arange(48000)/24000).astype(np.float32)*.02
            wavfile.write(reference,44100,resample_poly(tone,147,80).astype(np.float32))
            given=root/'given_raw.wav'
            wavfile.write(given,24000,tone)
            def info(path):
                rate,data=wavfile.read(path)
                return types.SimpleNamespace(samplerate=rate,duration=len(data)/rate)
            def read(path,**kwargs):
                rate,data=wavfile.read(path)
                return data,rate
            clone=types.SimpleNamespace(load_xtts=lambda device:(None,None),
                compute_latents=lambda *a,**kw:(None,None),
                generate=lambda model,text,lang,lat,path,**kw:wavfile.write(path,24000,tone))
            last,quiet=[None],[]
            def post(src,dst,blk,*a):
                last[0]=dict(blk, _source=Path(src).name);shutil.copyfile(src,dst)
            def loud_post(src,dst,blk,*a):
                if post_error and post_error(blk,Path(dst).name):
                    raise RuntimeError('render failed')
                post(src,dst,blk)
            def quiet_post(src,dst,blk,*a):
                quiet.append(dict(blk));post(src,dst,blk)
            class Encoder:
                def __init__(self,**kw): pass
                def embed(self,source,sr=None): return np.array([1.,0.])
                def cosine(self,a,b): return identity(last[0])
            modules={'xtts_clone':clone,'torch':types.SimpleNamespace(),
                'soundfile':types.SimpleNamespace(info=info,read=read),
                'speaker_identity':types.SimpleNamespace(SpeakerEncoder=Encoder)}
            if presets is not None:
                modules['voice_presets']=types.SimpleNamespace(PRESET_DIR=str(root),
                    name_from_reference=lambda path:'Voice',
                    save=lambda name,xtts_block,audio_block,**kw:presets.append(audio_block) or name)
            argv=['voice_comparator.py',str(reference),'FR','--xtts-block',settings.format_block({'seed':0}),
                '--audio-block',block,'--raw-clone',str(given),'--text','Bonjour Elo.',
                '--reference-interval','0.5','1.5','--device','cpu','--no-fit-dynamics','--no-identity-check',
                '--output',str(root/'base.wav'),'--output-optimised',str(root/'fit.wav'),
                '--output-raw',str(root/'raw.wav'),*extra]
            if dynamics:
                argv.remove('--no-fit-dynamics')
            out=io.StringIO()
            with patch.dict(sys.modules,modules),patch.object(sys,'argv',argv), \
                 patch.object(comparator,'apply_post',side_effect=loud_post), \
                 patch.object(comparator,'_quiet_post',side_effect=quiet_post), \
                 contextlib.redirect_stdout(out):
                self.assertFalse(comparator.main())     # exit status 0
            if return_report:
                report=json.loads((root/'fit_audio_search.json').read_text())
                for paths in report['validation_renders']:
                    self.assertTrue(all(Path(path).is_file() for path in paths.values()))
                return out.getvalue(),quiet,report
            return out.getvalue(),quiet

    def test_all_zero_effects_and_switches_receive_real_trials(self):
        out,quiet=self.run_main('[1, FR, 0.9, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 1]',
            ['--optimise-audio','nelder','--optimise-budget','80'],
            identity=lambda block:.7+.1*(block.get('lp')==8000))
        self.assertNotIn('Audio search failed',out)
        self.assertIn('ADOPTED',out)
        self.assertIn('lp',out)
        for field in ('lp','hp','NR','comp','de-ess','reverb','noise_gate'):
            self.assertGreater(len({block.get(field) for block in quiet}),1,field)
        # The pace, the stereo placement and the clipping protection are not
        # ECAPA's to judge: every render keeps them.
        self.assertEqual({(block['speed'],block['pan'],block['limiter']) for block in quiet},
                         {(.9,0,1)})
        self.assertRegex(out,r'speed: held at 0\.9 \(meditation pace')
        self.assertIn('held-out clone 2:',out)
        self.assertIn('Audio-search report:',out)

    def test_speed_pan_and_limiter_are_searched_on_request_only(self):
        _,quiet=self.run_main('[1, FR, 0.9, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 1]',
            ['--optimise-audio','nelder','--optimise-budget','80','--search-all-fields'],
            identity=lambda block:.7+.1*(block.get('lp')==8000))
        for field in ('speed','pan'):
            self.assertGreater(len({block.get(field) for block in quiet}),1,field)
        self.assertEqual({block['limiter'] for block in quiet},{0,1})

    def test_a_gain_elsewhere_only_changes_its_own_field(self):
        # Every render differs a little: the encoder sees each distinct block
        # through a small deterministic perturbation, as real renders do.
        def identity(block):
            digest=hashlib.sha256(repr(sorted((k,v) for k,v in block.items()
                                              if k!='_source')).encode()).digest()
            return .7+.1*(block.get('lp')==8000)+3e-4*(digest[0]/255-.5)
        out,_,report=self.run_main('[1, FR, 0.9, 0, 0, 0, 0, 90, 0, 0, 0.2, 0.15, 0, 0, 0, 1]',
            ['--optimise-audio','nelder','--optimise-budget','120'],identity,return_report=True)
        self.assertIn('ADOPTED',out)
        start,found,final=(report[k] for k in ('baseline_block','search_best_block','final_block'))
        moved=lambda block:{k for k in AUDIO_FIELDS if block[k]!=start[k]}
        self.assertGreater(len(moved(found)),1)       # the search alone let others drift
        self.assertEqual(moved(final),{'lp'})
        self.assertIn('pruned back to their start:',out)

    def test_a_failed_search_keeps_the_tone_fit_and_the_run_succeeds(self):
        block='[1, FR, 1, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 1]'
        tone,_=self.run_main(block,[])
        def lost(block):
            raise RuntimeError('encoder lost')
        presets=[]
        out,_,report=self.run_main(block,['--optimise-audio','nelder','--optimise-budget','80',
                                          '--save-preset'],
                                   lost,return_report=True,presets=presets)
        self.assertIn('[!] Audio search failed: encoder lost',out)
        self.assertIn('[OK] Done.',out)
        nxt=re.findall(r'Next \[\]\s*:\s*(.*)',out)
        self.assertEqual(nxt,re.findall(r'Next \[\]\s*:\s*(.*)',tone))
        self.assertEqual(report['error'],'encoder lost')
        # The tone fit is a complete result: saved like one from a run without the search.
        self.assertEqual(presets,[nxt[0].strip()])

    def test_a_failure_after_adoption_still_leaves_the_tone_fit(self):
        out,_=self.run_main('[1, FR, 1, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 1]',
            ['--optimise-audio','nelder','--optimise-budget','80'],
            identity=lambda block:.7+.1*(block.get('lp')==8000),
            post_error=lambda block,name:name=='cand.wav' and block.get('lp')==8000)
        self.assertIn('ADOPTED',out)
        self.assertIn('[!] Audio search failed: render failed',out)
        nxt=re.search(r'Next \[\]\s*:\s*\[([^\]]*)\]',out).group(1).split(',')
        self.assertEqual(float(nxt[8]),0)             # lp of the tone fit, not 8000
        self.assertIn('tone-fitted []',out)

    def test_one_regressing_holdout_rejects_the_candidate_and_retains_tested_zeros(self):
        def identity(block):
            direction=-1 if 'holdout_2' in block['_source'] else 1
            return .7 + direction*.1*(block.get('lp')==8000)
        out,_,report=self.run_main('[1, FR, 1, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 1]',
            ['--optimise-audio','nelder','--optimise-budget','80'], identity, return_report=True)
        self.assertIn('REJECTED',out)
        self.assertFalse(report['adopted'])
        self.assertEqual(report['candidate_block']['lp'],8000)
        self.assertEqual(report['final_block']['lp'],0)
        self.assertEqual(report['validation_evaluations'],4)
        self.assertEqual(len(report['validation_renders']),2)

    def test_a_holdout_that_regresses_rejects_even_with_a_good_mean(self):
        # +0.2 on one sentence, -0.01 on the other: the mean clears the bar,
        # the candidate still made one sentence worse.
        def identity(block):
            gain=.1*(block.get('lp')==8000)
            if 'holdout_1' in block['_source']:
                gain*=2
            elif 'holdout_2' in block['_source']:
                gain*=-.1
            return .7+gain
        out,_,report=self.run_main('[1, FR, 1, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 1]',
            ['--optimise-audio','nelder','--optimise-budget','80'],identity,return_report=True)
        self.assertGreater(np.mean([row['gain'] for row in report['validation']]),.02)
        self.assertIn('REJECTED: insufficient independent gains',out)
        self.assertEqual(report['final_block']['lp'],0)

    def test_the_dynamics_fit_says_it_keeps_the_given_values(self):
        # The clone and the reference are the same tone: both gaps are within
        # tolerance, and the block's own de-ess and comp are kept, not zeroed.
        out,_=self.run_main('[1, FR, 1, 0, 0, 0, 0, 90, 0, 0, 0.2, 0.3, 0, 0, 0, 1]',[],dynamics=True)
        self.assertRegex(out,r'de-ess: gap [+-]\d\.\d dB within tolerance \(1\.5\); keeping 0\.3')
        self.assertRegex(out,r'comp: gap [+-]\d\.\d dB within tolerance \(2\.0\); keeping 0\.2')
        self.assertNotIn('-> 0\n',out)
        nxt=re.search(r'Next \[\]\s*:\s*\[([^\]]*)\]',out).group(1).split(',')
        self.assertEqual((float(nxt[10]),float(nxt[11])),(.2,.3))

    def test_the_screening_tries_the_lowpass_off(self):
        out,quiet=self.run_main('[1, FR, 1, 0, 0, 0, 0, 90, 9000, 0, 0, 0, 0, 0, 0, 1]',['--screen-audio'],
            identity=lambda block:.8+(.05 if not block.get('lp') else 0))
        self.assertIn(0,[block['lp'] for block in quiet])
        self.assertRegex(out,r'lp\s+0\.0500\s+off\s+0\.8500')

    def test_the_identity_search_tries_the_lowpass_off_when_one_is_on(self):
        # Also the run that stopped on a NameError with --no-fit-dynamics.
        out,_=self.run_main('[1, FR, 1, 0, 0, 0, 0, 90, 9000, 0, 0, 0, 0, 0, 0, 1]',['--fit-identity'])
        self.assertIn('low-pass off',out)
        self.assertNotIn('Identity search unavailable',out)
        out,_=self.run_main('[1, FR, 1, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 1]',['--fit-identity'])
        self.assertNotIn('low-pass off',out)


if __name__=='__main__': unittest.main()
