"""Conservative voice routing, distinct from waveform source separation.

Female/male are user-facing output names. Pitch labels are heuristics; reference
labels designate the supplied speakers, independently of their pitch.
"""
from dataclasses import asdict, dataclass
import json
import math
import os
import numpy as np
from audio_checks import mono_audio


@dataclass
class Region:
    start: float
    end: float
    label: str
    reason: str = ''
    score: float | None = None
    speaker: str | None = None


def pitch_label(values, threshold=165., margin=12.):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 3:
        return 'unknown', None
    median = float(np.median(values))
    if median >= threshold + margin:
        return 'female', median
    if median <= threshold - margin:
        return 'male', median
    return 'unknown', median


def analysis_cells(segments, window=1.5, step=.5):
    """Nonoverlapping output cells with surrounding analysis context.

    Context never extends beyond the speech region. Cells are not concatenated
    with per-cell fades: adjoining cells of one output are merged on export.
    """
    if not (math.isfinite(window) and math.isfinite(step) and 0 < step <= window):
        raise ValueError('Require 0 < analysis step <= window')
    cells, contexts = [], []
    for start, end in segments:
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError('Invalid speech interval')
        count = max(1, math.ceil((end - start) / step))
        edges = np.linspace(start, end, count + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            center = (a + b) / 2
            left = max(start, min(center - window / 2, end - window))
            right = min(end, left + window)
            cells.append((float(a), float(b)))
            contexts.append((float(left), float(right)))
    return cells, contexts


class ReferenceMatcher:
    """Match each window to supplied speaker examples; never guess the other sex."""
    def __init__(self, encoder, references, minimum=.35, margin=.08):
        if not references or not 0 <= minimum <= 1 or not 0 <= margin <= 2:
            raise ValueError('References required; minimum in [0,1], margin in [0,2]')
        self.encoder, self.minimum, self.margin = encoder, minimum, margin
        self.references = {}
        for label, path in references.items():
            if label not in ('female', 'male'):
                raise ValueError('Unknown reference output name')
            self.references[label] = self._unit(encoder.embed(os.fspath(path)))
        if len(self.references) == 2:
            similarity = float(np.dot(*self.references.values()))
            if similarity >= .95:
                raise ValueError('References are too similar; supply distinct solo speakers')

    @staticmethod
    def _unit(embedding):
        embedding = np.asarray(embedding, dtype=float).ravel()
        norm = np.linalg.norm(embedding)
        if not embedding.size or not np.isfinite(embedding).all() or norm < 1e-12:
            raise ValueError('Invalid speaker embedding')
        return embedding / norm

    def match_embedding(self, embedding):
        embedding = self._unit(embedding)
        scores = sorted(((float(np.dot(embedding, ref)), label)
                         for label, ref in self.references.items()), reverse=True)
        score, label = scores[0]
        if score < self.minimum:
            return 'unknown', score, 'low_reference_similarity'
        if len(scores) > 1 and score - scores[1][0] < self.margin:
            return 'unknown', score, 'ambiguous_reference_match'
        return label, score, 'reference_match'

    def match(self, audio, sr):
        return self.match_embedding(self.encoder.embed(audio, sr=sr))


def pitch_regions(y, sr, segments, estimate, threshold=165., margin=12.):
    """Label short parts of a pitch track, rather than one whole long turn.

    A wide pitch range is not evidence of simultaneous speakers. Uncertain
    estimates go to review. Median filtering suppresses isolated octave jumps.
    """
    from scipy.ndimage import median_filter
    regions = []
    for start, end in segments:
        # Bound pYIN memory/time on continuous recordings.
        count = max(1, math.ceil((end - start) / 15.))
        for left, right in zip(np.linspace(start, end, count + 1)[:-1], np.linspace(start, end, count + 1)[1:]):
            chunk = y[int(left * sr):int(right * sr)]
            if not len(chunk) or np.max(np.abs(chunk), initial=0) < 1e-5:
                regions.append(Region(float(left), float(right), 'unknown', 'no_pitch'))
                continue
            f0, voiced, ratio, hop = estimate(chunk, sr)
            values = np.asarray(f0, dtype=float).copy()
            valid = np.asarray(voiced, dtype=bool) & np.isfinite(values) & (values > 0)
            if ratio < .12 or valid.sum() < 3:
                regions.append(Region(float(left), float(right), 'unknown', 'no_pitch'))
                continue
            # Smooth valid pitch values only; invalid frames do not become votes.
            filtered = np.full(len(values), np.nan)
            filtered[valid] = median_filter(values[valid], size=3, mode='nearest')
            cell_count = max(1, math.ceil((right - left) / .2))
            edges = np.linspace(left, right, cell_count + 1)
            for a, b in zip(edges[:-1], edges[1:]):
                i = max(0, int((a - left) * sr / hop))
                j = min(len(values), max(i + 1, math.ceil((b - left) * sr / hop)))
                label, median = pitch_label(filtered[i:j], threshold, margin)
                regions.append(Region(float(a), float(b), label, 'pitch_heuristic', median))
    # Brief unvoiced consonants may inherit only if both neighbours agree.
    for index in range(1, len(regions) - 1):
        previous, current, following = regions[index-1:index+2]
        if (current.label == 'unknown' and current.score is None and current.end-current.start <= .22
                and previous.label == following.label and previous.label != 'unknown'
                and abs(previous.end-current.start) < 1e-6 and abs(current.end-following.start) < 1e-6):
            current.label, current.reason = previous.label, 'short_unvoiced_bridge'
    return regions


def embedding_regions(y, sr, segments, encoder, estimate, references=None,
                      minimum=.35, margin=.08, threshold=165., pitch_margin=12.):
    cells, contexts = analysis_cells(segments)
    matcher = ReferenceMatcher(encoder, references, minimum, margin) if references else None
    regions, embeddings, valid_indices = [], [], []
    for (start, end), (left, right) in zip(cells, contexts):
        chunk = y[int(left * sr):int(right * sr)]
        region = Region(start, end, 'unknown', 'short_or_silent_context')
        regions.append(region)
        if right-left < .5 or np.sqrt(np.mean(chunk.astype(float)**2)) < 1e-5:
            continue
        embedding = ReferenceMatcher._unit(encoder.embed(chunk, sr=sr))
        if matcher:
            region.label, region.score, region.reason = matcher.match_embedding(embedding)
        else:
            embeddings.append(embedding)
            valid_indices.append(len(regions)-1)
    if matcher or not embeddings:
        return regions
    # Variable cluster count: a single speaker must not be forced into two sexes.
    from scipy.cluster.hierarchy import linkage, fcluster
    E = np.stack(embeddings)
    clusters = (np.ones(len(E), dtype=int) if len(E) < 2 else
                fcluster(linkage(E, method='average', metric='cosine'), t=.3, criterion='distance'))
    for cluster in sorted(set(clusters)):
        members = np.flatnonzero(clusters == cluster)
        center = ReferenceMatcher._unit(E[members].mean(axis=0))
        ranked = sorted(members, key=lambda index:-float(E[index] @ center))
        pitches = []
        for index in ranked[:6]:
            left, right = contexts[valid_indices[index]]
            values, voiced, _, _ = estimate(y[int(left*sr):int(right*sr)], sr)
            values = np.asarray(values, float)
            valid = np.asarray(voiced, bool) & np.isfinite(values)
            if valid.sum() >= 3:
                pitches.append(float(np.median(values[valid])))
        # Aggregate one statistic per window, avoiding duration-based domination.
        label, median = pitch_label(np.repeat(pitches, 3), threshold, pitch_margin)
        for index in members:
            region = regions[valid_indices[index]]
            region.label, region.score = label, median
            region.reason, region.speaker = 'speaker_pitch_heuristic', f'speaker_{cluster}'
    return regions


def partition_turns(turns, speaker_labels):
    """Union duplicate diarization turns; mark simultaneous distinct speakers."""
    events = {}
    for start, end, speaker in turns:
        if start < 0 or end <= start or not math.isfinite(start + end):
            raise ValueError('Invalid diarization turn')
        events.setdefault(start, []).append((speaker, 1))
        events.setdefault(end, []).append((speaker, -1))
    active, result, previous = {}, [], None
    for time in sorted(events):
        speakers = [speaker for speaker, count in active.items() if count > 0]
        if previous is not None and time > previous and speakers:
            label = speaker_labels.get(speakers[0], 'unknown') if len(speakers) == 1 else 'overlap'
            result.append(Region(previous, time, label, 'diarization',
                                 speaker=speakers[0] if len(speakers) == 1 else None))
        for speaker, delta in events[time]:
            active[speaker] = active.get(speaker, 0) + delta
        previous = time
    return result


def assemble(y, sr, regions, labels, silence='auto', minimum=0., fade_ms=5.):
    intervals = sorted((max(0, round(r.start*sr)), min(len(y), round(r.end*sr)))
                       for r in regions if r.label in labels)
    merged = []
    for start, end in intervals:
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    # Minimum applies to merged same-label runs, not arbitrary analysis cells.
    merged = [(s,e) for s,e in merged if (e-s)/sr >= minimum]
    if not merged:
        return None
    output = np.zeros_like(y) if silence == 'auto' else []
    for start, end in merged:
        chunk = y[start:end].copy()
        fade = min(round(fade_ms*sr/1000), len(chunk)//4)
        if fade > 1:
            ramp = np.linspace(0,1,fade)
            chunk[:fade] *= ramp
            chunk[-fade:] *= ramp[::-1]
        if silence == 'auto':
            output[start:end] = chunk
        else:
            if output and float(silence) > 0:
                output.append(np.zeros(round(float(silence)*sr), dtype=y.dtype))
            output.append(chunk)
    return output if silence == 'auto' else np.concatenate(output)


def write_report(path, regions, method, references=None):
    counts = {}
    for label in sorted({region.label for region in regions}):
        intervals=sorted((r.start,r.end) for r in regions if r.label==label)
        merged=[]
        for start,end in intervals:
            if merged and start<=merged[-1][1]:
                merged[-1][1]=max(end,merged[-1][1])
            else:
                merged.append([start,end])
        counts[label]=sum(end-start for start,end in merged)
    body = dict(method=method, references=references or {}, durations=counts,
                note='Pitch labels and matching thresholds are heuristics, not calibrated confidence. Overlap is only detected by diarization, not by pitch range.',
                regions=[asdict(r) for r in regions])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(body, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return body
