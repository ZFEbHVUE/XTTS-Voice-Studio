"""Voice presets that say more than their two blocks (3.8.0).

A preset file is two settings lines and commentary. The commentary may be as
long as it likes, but it must never change what the file means: pasted whole
into a script (the generator drops '#' lines) it still gives exactly the two
blocks, and no value, however odd, can smuggle a third one in.

    python -m unittest test_preset_info_regressions -v
"""
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import voice_presets as VP
import xtts_pipeline as pipeline
import voice_comparator as comparator
import xtts_settings as settings

XB = '{1, 100, 0, 200, 100, 250, 0.64, 50, 0.85, 5, 1, 45, 4, 0, 1, 1, 45}'
AB = '[1, FR, 1, -1, -4, 4, 1.5, 95, 0, 0, 0, 0, 0, 0, 0, 1]'


def without_comments(text):
    """What the generator reads: the file minus its '#' lines."""
    return [l.strip() for l in text.splitlines() if l.strip() and not l.lstrip().startswith('#')]


class Dir(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        patcher = patch.object(VP, 'PRESET_DIR', self._td.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.dir = Path(self._td.name)


class ExplainTests(unittest.TestCase):
    def test_xtts_block_in_words(self):
        text = '\n'.join(VP.explain_xtts(XB))
        for part in ('voice 1', 'seed 100', 'temp 0.64', 'top_k 50', 'rep_pen 5',
                     'trim_end 200 ms', 'fade_out 250 ms', 'gpt_cond_len 45 s',
                     'gpt_cond_chunk_len 4 s', 'max_ref_len 45 s',
                     'sound_norm_refs off', 'num_beams 1', 'do_sample on'):
            self.assertIn(part, text)

    def test_short_block_does_not_invent_settings(self):
        text = '\n'.join(VP.explain_xtts('{2, 7, 0, 0, 100, 250, 0.7}'))
        self.assertIn('seed 7', text)
        self.assertIn('top_k not set', text)            # left out: the voice keeps its own
        self.assertIn('max_ref_len not set', text)

    def test_zero_placeholders_and_unseeded(self):
        text = '\n'.join(VP.explain_xtts('{1, none, 0, 0, 0, 0, 0, 50, 0.85}'))
        self.assertIn('seed none (unseeded)', text)
        self.assertIn('temp inherited (0)', text)        # 0 is the placeholder, not a setting
        self.assertIn('trim_start 0 ms', text)           # for trims 0 is a real value

    def test_audio_block_in_words(self):
        text = '\n'.join(VP.explain_audio(AB))
        for part in ('language FR', 'speed 1', 'volume -1 dB', 'EQ low -4 dB',
                     'mid +4 dB', 'high +1.5 dB', 'high-pass 95 Hz', 'low-pass off',
                     'noise reduction off', 'compression off', 'de-esser off',
                     'reverb off', 'noise gate off', 'pan centre', 'limiter on'):
            self.assertIn(part, text)

    def test_audio_values_with_effects(self):
        text = '\n'.join(VP.explain_audio('[2, en, 0.9, 6.471, 0.1, 4, 0.6, 90, 8500, 0.5, 0.4, 0.3, 0.2, -40, -0.5, 0]'))
        for part in ('language EN', 'speed 0.9', 'volume +6.471 dB', 'low-pass 8500 Hz',
                     'noise reduction 0.5', 'compression 0.4', 'de-esser 0.3',
                     'reverb 0.2', 'noise gate -40 dB', 'pan left 0.5', 'limiter off'):
            self.assertIn(part, text)

    def test_audio_block_without_language_or_values(self):
        text = '\n'.join(VP.explain_audio('[3, 0.9, 2]'))
        self.assertIn('voice     : 3 | language default', text)
        self.assertIn('speed 0.9', text)
        self.assertIn('volume +2 dB', text)
        self.assertIn('high-pass default', text)         # left out: not shown as off

    def test_unreadable_blocks_give_nothing(self):
        for bad in ('', 'garbage', '{', '[x]', None):
            self.assertEqual(VP.explain_xtts(bad), [])
            self.assertEqual(VP.explain_audio(bad), [])


class FileTests(Dir):
    def save(self, **kw):
        return VP.save('Voice', XB, AB, **kw)

    def test_blocks_stay_at_the_top_and_the_commentary_below(self):
        name = self.save(source='pipeline', acoustics='female | F0 200 Hz',
                         info={'Run': ['seeds 1 2 3']})
        lines = (self.dir / (name + '.txt')).read_text(encoding='utf-8').splitlines()
        first_block = next(i for i, l in enumerate(lines) if l.startswith('{'))
        self.assertLess(first_block, 12)
        self.assertEqual(lines[first_block + 1], AB)
        self.assertTrue(all(l.startswith('#') or not l.strip() for l in lines[first_block + 2:]))

    def test_pasted_whole_into_a_script_gives_exactly_the_two_blocks(self):
        name = self.save(info={'Run': ['seeds 1 2 3'], 'Files': ['a.wav']})
        text = (self.dir / (name + '.txt')).read_text(encoding='utf-8')
        self.assertEqual(without_comments(text), [XB, AB])

    def test_commentary_is_always_written_and_read_back(self):
        name = self.save(info={'Run': ['seeds 1 2 3', 'budget 400']})
        entry = VP.read_file(VP.path_of(name))
        self.assertEqual((entry['xtts'], entry['audio']), (XB, AB))
        self.assertEqual(entry['info']['Run'], ['seeds 1 2 3', 'budget 400'])
        words = ' '.join(entry['info']['In words'])
        self.assertIn('{} sampling', words)
        self.assertIn('[] filters', words)

    def test_empty_sections_are_not_written(self):
        name = self.save(info={'Run': [], 'Files': None, 'Curation': ['x']})
        entry = VP.read_file(VP.path_of(name))
        self.assertEqual(set(entry['info']), {'In words', 'Curation'})

    def test_values_cannot_break_out_of_the_comment(self):
        evil = 'x\n{9, 9, 9}\n[9, FR, 9]\nBonjour'
        name = VP.save(evil, XB, AB, reference=evil, source=evil, note=evil,
                       acoustics=evil, info={evil: [evil, 'a\r\n{8}']})
        raw = VP.path_of(name)
        text = Path(raw).read_text(encoding='utf-8')
        self.assertEqual(without_comments(text), [XB, AB])
        entry = VP.read_file(raw)
        self.assertEqual((entry['xtts'], entry['audio']), (XB, AB))

    def test_a_section_ends_at_the_first_line_that_is_not_a_comment(self):
        # A hand-edited file: comments, a block pair, then a stray comment.
        (self.dir / 'Hand.txt').write_text(
            f"# ---- Run ----\n#   seeds 1 2\n{XB}\n{AB}\n# stray remark\n", encoding='utf-8')
        e = VP.get('Hand')
        self.assertEqual(e['info'], {'Run': ['seeds 1 2']})

    def test_header_keys_survive(self):
        name = self.save(reference='r.wav', source='pipeline', scores={'identity': .6},
                         note='n', acoustics='a | b', references=['/x/y.wav'])
        e = VP.read_file(VP.path_of(name))
        self.assertEqual((e['reference'], e['source'], e['note'], e['acoustics']),
                         ('r.wav', 'pipeline', 'n', 'a | b'))
        self.assertEqual(e['references'], ['/x/y.wav'])
        self.assertIn('identity 0.600', e['scores_text'])
        self.assertIn('a | b', VP.describe(name))

    def test_commentary_headings_cannot_be_mistaken_for_header_keys(self):
        # A section called 'Voice' or 'Source' must not overwrite the header's.
        name = self.save(acoustics='the real voice', source='pipeline',
                         info={'Voice': ['other : thing'], 'Source': ['Source : other']})
        e = VP.read_file(VP.path_of(name))
        self.assertEqual(e['acoustics'], 'the real voice')
        self.assertEqual(e['source'], 'pipeline')

    def test_old_files_still_read(self):
        old = ("# Voice preset: Old\n# Source    : pipeline  |  2026-07-29 23:16\n"
               "# Reference : r.wav\n# Scores    : held-out 0.812 | identity 0.725\n"
               "# Paste these two lines into a script (the '#' lines are ignored).\n"
               f"{XB}\n{AB}\n")
        (self.dir / 'Old.txt').write_text(old, encoding='utf-8')
        e = VP.get('Old')
        self.assertEqual((e['xtts'], e['audio'], e['source'], e['date']),
                         (XB, AB, 'pipeline', '2026-07-29 23:16'))
        self.assertNotIn('info', e)
        self.assertIn('held-out 0.812', VP.describe('Old'))
        self.assertIn('Reference : r.wav', VP.info_text('Old'))

    def test_overwrite_and_history_names_still_work(self):
        a = self.save()
        b = VP.save('Voice', XB, AB, overwrite=False)
        self.assertEqual((a, b), ('Voice', 'Voice (2)'))

    def test_info_text_is_the_commentary_without_the_paste_hint(self):
        name = self.save(info={'Run': ['seeds 1 2 3']})
        text = VP.info_text(name)
        self.assertIn('Voice preset: Voice', text)
        self.assertIn('---- In words', text)
        self.assertIn('seeds 1 2 3', text)
        self.assertNotIn('Paste these two lines', text)
        self.assertFalse(any(l.startswith('#') for l in text.splitlines()))
        self.assertEqual(VP.info_text('missing'), '')

    def test_insert_text_names_the_voice_above_the_blocks(self):
        name = self.save(acoustics='female | F0 200 Hz')
        text = VP.insert_text(VP.get(name))
        lines = text.splitlines()
        self.assertEqual(lines[0], '# Voice - female | F0 200 Hz')
        self.assertEqual(without_comments(text), [XB, AB])
        # Without acoustics the source stands in; without either, just the name.
        self.assertEqual(VP.insert_text(dict(name='V', source='gui', xtts=XB, audio=AB)).splitlines()[0], '# V - gui')
        self.assertEqual(VP.insert_text(dict(name='V', xtts=XB, audio=AB)).splitlines()[0], '# V')
        # A name holding a line break stays one comment line.
        text = VP.insert_text(dict(name='a\n{3}', xtts=XB, audio=AB))
        self.assertEqual(without_comments(text), [XB, AB])

    def test_legacy_json_migration_writes_the_new_layout(self):
        legacy = self.dir / 'legacy.json'
        legacy.write_text(json.dumps({'Old': dict(xtts=XB, audio=AB, source='imported')}))
        with patch.object(VP, '_LEGACY_JSON', str(legacy)), contextlib.redirect_stdout(io.StringIO()):
            self.assertIn('Old', VP.load())
        self.assertEqual(without_comments((self.dir / 'Old.txt').read_text(encoding='utf-8')), [XB, AB])


class PipelineFixture(Dir):
    """The pipeline run against stand-ins for its four tools, saving through
    the real voice_presets into a temporary folder."""

    report = None

    def run_pipeline(self, extra=(), search_report=None, search_in_stdout=True,
                     breaking=None, curated=False, curate_new=False, comparator_out='',
                     optimiser_extra=None):
        td = self.dir
        self.commands = []
        ref, other = td / 'a.wav', td / 'b.wav'
        ref.touch(); other.touch()
        xb = settings.format_block(dict(seed=100, temp=.64, max_ref_len=45))
        ab = '[1, FR, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]'
        fitted = '[1, FR, 1, -1, -4, 4, 1.5, 95, 0, 0, 0, 0, 0, 0, 0, 1]'
        if search_report is not None:
            for name in ('a', 'blend'):
                (td / f'audio_search_{name}.json').write_text(json.dumps(search_report), encoding='utf-8')
        if curated:
            (td / 'a_curated.wav').touch()
        def run(cmd, tag):
            tool = os.path.basename(cmd[1])
            self.commands.append(list(cmd))
            if tool == 'curate_reference.py':
                Path(cmd[cmd.index('-o') + 1]).touch()
                return 0, ''
            if tool == 'voice_analyser.py':
                return 0, ('Voice type     : female\nF0 median      : 200 Hz\nVoiced ratio   : 61%\n'
                           'RMS level      : -23.0 dBFS\nEstimated SNR  : 30 dB\n' + xb + '\n' + ab)
            if tool == 'xtts_optimize.py':
                Path(cmd[cmd.index('--result-json') + 1]).write_text(json.dumps(dict(
                    schema_version=1, xtts_block=xb, selected_refs=[str(other)],
                    scores=dict(held_out=.8, identity=.7),
                    score_definition=dict(objective='identity', w_accent=.6, w_identity=.4),
                    generations=dict(conditioning=34, sampling=40, holdout=6),
                    truncated_stages={'sampling': 'cut'},
                    conditioning_confirmation=dict(finalists=[dict(mean=.71)], base_mean=.70,
                        seeds=[1, 2], adopted=False, base='gpt30', bar=.02, winner=None),
                    holdout_detail=[dict(text='Une phrase inconnue.', accent=.9, identity=.62,
                                         score=.74, wer=.0, wav=str(td / 'h1.wav'))],
                    holdout_alerts=[dict(sentence=1, reason='slow')], **(optimiser_extra or {}))))
                return 0, 'x'
            weighting = cmd[cmd.index('--eq-weighting') + 1] if '--eq-weighting' in cmd else 'a'
            out = ('[*] Fit input: ' + json.dumps({'text': 'Bonjour a tous, ceci est la phrase.',
                                                   'interval': [0.0, 14.98]}) + '\n')
            if search_in_stdout:
                out += f'[*] Audio-search report: {td / ("audio_search_" + weighting + ".json")}\n'
            return 0, out + comparator_out + 'Next []    : ' + fitted
        if breaking is not None:
            patcher = patch.object(pipeline, '_preset_info', side_effect=breaking)
            patcher.start(); self.addCleanup(patcher.stop)
        out = io.StringIO()
        with patch.object(pipeline, 'run_stream', run), contextlib.redirect_stdout(out), \
             patch.object(sys, 'argv', ['xtts_pipeline.py', '--voice', str(ref), 'FR',
                                        *([] if (curated or curate_new) else ['--no-curate']),
                                        *extra]):
            pipeline.main()
        return out.getvalue(), fitted, xb


class PipelinePresetTests(PipelineFixture):
    def test_pipeline_preset_carries_the_whole_run(self):
        out, fitted, xb = self.run_pipeline()
        e = VP.get('a')
        self.assertEqual((e['xtts'], e['audio']), (xb, fitted))
        info = e['info']
        self.assertEqual(set(info), {'In words', 'Run', 'Curation', 'Starting point (analyser priors)',
                                     'Conditioning', 'Tone fit', 'Audio search',
                                     'Unseen sentences (hold-out)', 'Files'})
        run = ' '.join(info['Run'])
        self.assertIn('objective identity', run)
        self.assertIn('w_accent 0.6', run)
        self.assertIn('sampling 40', run)
        self.assertIn('stages cut short', run)
        self.assertIn('skipped (--no-curate)', ' '.join(info['Curation']))
        self.assertIn(xb, ' '.join(info['Starting point (analyser priors)']))
        self.assertIn('kept the base gpt30', ' '.join(info['Conditioning']))
        tone = ' '.join(info['Tone fit'])
        self.assertIn("EQ weighting 'a'", tone)
        self.assertIn('Bonjour a tous', tone)
        self.assertIn('0.00 s to 14.98 s', tone)
        self.assertIn('off', ' '.join(info['Audio search']))
        unseen = ' '.join(info['Unseen sentences (hold-out)'])
        self.assertIn('identity 0.620', unseen)
        self.assertIn('Une phrase inconnue', unseen)
        self.assertIn('listen to it', unseen)
        files = ' '.join(info['Files'])
        self.assertIn('a_pipeline_clone.wav', files)
        self.assertIn('h1.wav', files)
        # and the file itself still means two blocks
        text = Path(VP.path_of('a')).read_text(encoding='utf-8')
        self.assertEqual(without_comments(text), [xb, fitted])

    def test_audio_search_outcome_is_written(self):
        for report, expected in (
                (dict(adopted=True, baseline_score=.6, best_score=.64, adoption_tolerance=.02), 'result: adopted'),
                (dict(adopted=False, baseline_score=.6, best_score=.6), 'rejected, the tone fit is kept'),
                (dict(error='ffmpeg failed'), 'FAILED: ffmpeg failed')):
            with self.subTest(expected=expected):
                self.run_pipeline(('--optimise-audio', 'nelder', '--audio-objective', 'identity'), search_report=report)
                audio = ' '.join(VP.get('a')['info']['Audio search'])
                self.assertIn(expected, audio)
                self.assertIn('method nelder', audio)
                self.assertIn('held', audio)             # speed, pan, limiter
                self.assertIn('audio_search_a.json', audio)

    def test_alternative_keeps_its_own_audio_search_report(self):
        self.run_pipeline(('--eq-weighting', 'a+blend', '--optimise-audio', 'nelder'),
                          search_report=dict(adopted=False, baseline_score=.6, best_score=.6))
        self.assertIn('audio_search_a.json', ' '.join(VP.get('a')['info']['Audio search']))
        alt = ' '.join(VP.get('a blend')['info']['Audio search'])
        self.assertIn('audio_search_blend.json', alt)
        self.assertNotIn('audio_search_a.json', alt)

    def test_an_existing_curated_file_is_said_to_be_reused(self):
        self.run_pipeline(curated=True)
        cur = ' '.join(VP.get('a')['info']['Curation'])
        self.assertIn('a_curated.wav', cur)
        self.assertIn('reused', cur)

    def test_curation_length_is_a_target_not_a_result(self):
        self.run_pipeline(curate_new=True)
        cur = ' '.join(VP.get('a')['info']['Curation'])
        self.assertIn('keep target 45 s', cur)
        self.assertNotIn('kept', cur)

    def test_audio_search_notes_say_what_the_criterion_is(self):
        self.run_pipeline(('--optimise-audio', 'nelder', '--audio-objective', 'identity'),
                          search_report=dict(adopted=True, baseline_score=.6, best_score=.64,
                                             adoption_tolerance=.02))
        audio = ' '.join(VP.get('a')['info']['Audio search'])
        self.assertIn('objective ECAPA identity only', audio)
        self.assertIn('{} objective applies to XTTS generation', audio)
        self.assertIn('heuristic threshold, not a statistical confidence interval', audio)

    def test_unseen_renders_are_said_to_be_raw(self):
        self.run_pipeline()
        files = ' '.join(VP.get('a')['info']['Files'])
        self.assertIn('h1.wav (raw XTTS, before the [])', files)

    def test_unreadable_search_report_is_said_not_hidden(self):
        self.run_pipeline(('--optimise-audio', 'nelder'))        # report path printed, file missing
        self.assertIn('no report read', ' '.join(VP.get('a')['info']['Audio search']))

    def test_alternative_weighting_gets_its_own_sections(self):
        self.run_pipeline(('--eq-weighting', 'a+blend'))
        main, alt = VP.get('a'), VP.get('a blend')
        self.assertIn("EQ weighting 'a'", ' '.join(main['info']['Tone fit']))
        self.assertIn("EQ weighting 'blend'", ' '.join(alt['info']['Tone fit']))
        self.assertIn("alternative to 'a'", ' '.join(alt['info']['Run']))
        self.assertIn('Bonjour a tous', ' '.join(alt['info']['Tone fit']))

    def test_a_failure_building_the_commentary_never_costs_the_preset(self):
        out, fitted, xb = self.run_pipeline(breaking=RuntimeError('boom'))
        self.assertIn('Preset information skipped: boom', out)
        e = VP.get('a')
        self.assertEqual((e['xtts'], e['audio']), (xb, fitted))
        self.assertIn('In words', e['info'])             # the blocks in words need no run data


class ComparatorInfoTests(unittest.TestCase):
    ARGS = types.SimpleNamespace(eq_weighting='blend', target_dbfs=-20., optimise_audio='nelder',
                                 optimise_budget=400)

    def test_comparator_info(self):
        info = comparator.preset_info(self.ARGS, 'Une   phrase.\n', (0., 14.98),
                                      dict(baseline_score=.6, best_score=.64), True, 'b.wav', 'o.wav')
        self.assertIn("EQ weighting 'blend'", info['Tone fit'][0])
        tone = ' '.join(info['Tone fit'])
        self.assertIn('-20 dBFS', tone)
        self.assertIn('fitted sentence: Une phrase.', tone)
        self.assertIn('0.00 s to 14.98 s', tone)
        audio = ' '.join(info['Audio search'])
        self.assertIn('result: adopted', audio)
        self.assertIn('objective ECAPA identity only', audio)
        self.assertIn('0.6000 -> 0.6400', audio)
        self.assertEqual(info['Files'], ['base clone      : b.wav', 'optimised clone : o.wav'])

    def test_comparator_info_without_search_and_on_failure(self):
        off = types.SimpleNamespace(eq_weighting='a', target_dbfs=None, optimise_audio='none')
        self.assertIn('off', comparator.preset_info(off, None, None, None, False)['Audio search'][0])
        failed = comparator.preset_info(self.ARGS, 't', None, dict(error='boom'), False)
        self.assertIn('FAILED: boom', ' '.join(failed['Audio search']))

    def test_comparator_info_never_raises(self):
        self.assertIsNone(comparator.preset_info(self.ARGS, 't', 'not an interval', None, False))


class AnalyserTests(Dir):
    def test_analyser_preset_says_what_was_measured(self):
        try:
            import voice_analyser as analyser
        except (SystemExit, ImportError):          # the analyser needs librosa
            self.skipTest('librosa is not installed')
        params = dict(trim_start=0, trim_end=200, fade_in=100, fade_out=250, temperature=.7,
                      top_k=50, top_p=.85, repetition_penalty=5, length_penalty=1,
                      gpt_cond_len=30, gpt_cond_chunk_len=4, sound_norm_refs=0, speed=1, volume=0,
                      eq_low=0, eq_mid=0, eq_high=0, highpass=95, lowpass=0, noise_reduction=0,
                      compression=0, deesser=0, reverb=0, noise_gate=0, pan=0, limiter=1)
        stats = dict(voice_type='female', f0_median=200., f0_jitter=.12, rms_db=-23., snr=30.,
                     crest_factor_db=14., voiced_ratio=.6, sib_ratio=.3, breathiness=.1)
        with contextlib.redirect_stdout(io.StringIO()):
            analyser.display_results(params, stats, voice_num=1, wav_file='v.wav', language='FR',
                                     save_preset=True, preset_name='Analysed')
        e = VP.get('Analysed')
        self.assertEqual(set(e['info']), {'In words', 'Analysis'})
        analysis = ' '.join(e['info']['Analysis'])
        self.assertIn('F0 median 200 Hz', analysis)
        self.assertIn('crest factor 14.0 dB', analysis)
        self.assertIn('priors only', analysis)
        self.assertEqual(e['note'], 'priors only - audio block not fitted yet')


if __name__ == '__main__':
    unittest.main()
