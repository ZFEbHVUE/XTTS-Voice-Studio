# Brainwave Studio

**by Stéphane "ZFEbHVUE"**

A generator of binaural, isochronic and monaural beats, plus a physically
modelled singing bowl. Runs standalone or as the **[Wav] Brainwave** tab of
XTTS Voice Studio.

> **Technical honesty.** This tool renders the signals accurately — a binaural
> beat is a real binaural beat, an isochronic pulse train is a real pulse
> train. What it **cannot** claim is the effect. See
> [What the evidence supports](#what-the-evidence-supports) before believing
> anything a meditation video tells you, this one included.

---

## What it produces

| Mode | How the beat is made | Headphones | Stereo needed |
|---|---|---|---|
| **Binaural** | Two close tones, one per ear; the beat is built by the brainstem | **required** | yes |
| **Isochronic** | The tone itself is switched on and off | no | no |
| **Monaural** | The two tones are mixed *before* the ear — a real acoustic beat | no | no |
| **Bowl** | Independent vibration modes with their own decays and beats | no | no |

**On speakers, binaural becomes monaural.** The two channels mix in the air, so
the phase difference the brainstem needs is destroyed. Measured: summing a
binaural render to mono gives monaural to within 10⁻¹⁶. If your listeners use
speakers or a mono Bluetooth device, use isochronic.

---

## The session model

Everything is built from **segments**. A session is a list of them, rendered
end to end with a crossfade at each join.

**The rule: everything you set on the left panel is captured by `+ Add`.**
Whatever `▶ Play` previews is exactly what the segment will store — mode,
carrier, beat, duration, duty, stereo offset, beat level, rotation, drone,
noise, ducking, tuning, and the segment's own music with its level.

### Building a session

1. Set the tone and its music on the left
2. **`▶ Play`** — you hear precisely what the segment will contain
3. **`+ Add`** — the segment stores all of it, and the music panel clears so
   the next segment does not inherit the file by accident
4. Repeat
5. **`♪ Set GLOBAL music`** — the bed that plays under the whole session, on
   top of whatever music individual segments carry

**Click a segment** to load all of it back for editing, then **`Update`**, or
**`− Del`** to remove it. `Copy (export)` / `Paste (import)` move a session in
and out as readable text.

The list shows only what is actually set:

```
bina | 400Hz | 6.0Hz | 600s | beat -20dB | rot 2.0/min | drone 0.15 | ♪ ocean.mp3 -4dB
bina | 400Hz | 4.0Hz | 600s
```

---

## Levels

Beat, per-segment music and global music all run from **0 to −80 dB**, with a
numeric entry beside each slider for exact values.

Why −80 and not −100: a 16-bit WAV has its noise floor at −96 dBFS, so below
about −90 the signal no longer exists in the file. −80 is already inaudible
while remaining representable.

### Why the beat sits far below the music

Commercial "theta meditation" tracks put the beat **18–25 dB under** the music,
which is why you never hear it as a test tone. The label shows the gap:

```
Beat level (dB)   [──●──────]  -22.0   under the music
Music level (dB)  [───────●─]   -3.0   (+19 dB vs beat)
```

Aim for **+15 to +25 dB** in favour of the music for that style. Put the
carrier on a note of the music (via `Tuning`) and the two fuse into a chord
instead of coexisting.

### `also: ☐ drone ☐ noise`

Beat level normally attenuates **only the beat**. Drone and noise keep their own
levels — so pushing the beat down leaves you hearing nothing but the drone, and
since the drone is the same in every mode, binaural / monaural / bowl all sound
alike. Measured, with beat at −20 dB, drone 0.30, noise 0.10:

| drone | noise | overall level |
|---|---|---|
| ☐ | ☐ | **−4.9 dB** — the beat vanishes under an unchanged drone |
| ☑ | ☐ | −15.5 dB |
| ☑ | ☑ | **−20.0 dB** — the balance you set is preserved |

Tick them to move the whole tone bed together: set your mix at 0 dB, then push
it under the music in one gesture.

Note the asymmetry — ticking **drone** changes almost everything, ticking
**noise** almost nothing. The drone is the real masker: it is tuned to the
carrier and sits on top of the beat.

---

## Rotation

A slow pan of the stereo image, in turns per minute, with a depth control.

Constant-power law (cos/sin): the two gains always satisfy `gl² + gr² = 1`,
measured at 0.0000 dB of variation. A linear pan would lose 3 dB in the middle
and pulse audibly at slow speeds.

- **0.5–3 turns/min** — gentle, hypnotic
- **above 4** — reads as an effect

The phase carries across segments, so two consecutive parts at the same speed
keep turning smoothly instead of snapping back to centre (verified to 4×10⁻¹⁶
against a continuous render).

### Stereo offset (isochronic only)

Shifts the *pulses* between the ears rather than panning. At 180° the channels
are fully decorrelated (measured correlation 0.000) and the sound seems to turn
inside the head — but a mono sum loses 3 dB, so it partly cancels on speakers.
90° is gentler and usually more musical.

This is a **spatial effect, not a stronger stimulus**. Nothing focuses inside
the skull: a 6 Hz modulation has a wavelength of about 57 m, far too long to be
focused in a 20 cm head. (Techniques that *do* reach deep structures —
temporal interference stimulation, focused ultrasound — use electric fields or
500 kHz carriers, not headphones.)

---

## The singing bowl

A real bowl is **not** a harmonic series. Its modes sit at frequencies that are
not integer multiples of the fundamental, each with its own decay, and — because
a hammered bowl is never perfectly circular — each mode splits into a close pair
whose interference gives that slow pulsing. So each mode has **its own beat
rate**.

### Editing a bowl

Select mode **Bowl**, then **`Edit bowl…`**. One mode per line:

```
# freq_hz  amp  decay_s  beat_hz
256   1.0   35   0.9
704   0.55  12   2.1
1326  0.32   5   3.4
```

Only the frequency is required — amplitude 1.0, a decay scaled from the
fundamental and no beat are assumed. A bare list of frequencies read off a
phone analyser is already usable.

Three starting presets (small / medium / large), **Check** to see what was
understood before applying, **Clear** to return to the ratio-based bowl.

### Measuring your own bowl

**`Analyse a WAV…`** extracts the modes from a recording and fills the table.

Method: the spectrum of the whole take gives the frequencies; each mode is then
band-pass filtered on its own and its envelope followed — the slope of
log(envelope) is the decay, and the envelope's residual ripple is that mode's
beat rate.

Validated against a synthetic bowl with known values:

| Real | Measured |
|---|---|
| 256.0 Hz, decay 20 s, beat 0.90 | **256.4 Hz, 22.2 s, 0.87** |
| 704.0 Hz, 8 s, 2.10 | **705.1 Hz, 7.9 s, 2.06** |
| 1326.0 Hz, 3.5 s, 3.40 | **1325.9 Hz, 3.5 s, 3.33** |

Frequencies to better than 0.2 %.

**For a good recording:** one strike, let it ring to the end, nothing else in
the file. The longer the tail, the more accurate the decay times.

---

### A sequence of chakras

**`Add all 7`**, beside the chakra buttons, appends one segment per chakra in
order, each at the frequency of the chosen tuning. Every other setting on the
left is kept, and the panel is put back where you left it afterwards.

**A sequence, not a sweep.** A bowl does not change pitch while it rings — its
frequency comes from its geometry — so a glissando would sound like a
speeded-up tape rather than like seven bowls. Each chakra is its own struck
segment that decays before the next.

#### With a measured bowl loaded

Measured modes override the carrier, so the chakra buttons would otherwise have
no effect. Instead the bowl is **transposed** onto each chakra: everything
scales by one factor, which is the same as playing bowls of different sizes from
the same workshop — what a set of tuned bowls is.

What is preserved, and what follows:

| | Behaviour |
|---|---|
| Ratios between modes | **unchanged** (verified identical to 0.002) |
| Mode frequencies | scale with the factor |
| Beat rates | scale too — the detuning is a fraction of the mode |
| Decays | scale as 1/k: a smaller bowl rings shorter |
| Amplitudes | untouched — they describe how it was struck |

A 256 Hz bowl across `A=440`:

```
Root     261.6Hz  decay 34.3s  beat 0.92
Sacral   293.7Hz  decay 30.5s  beat 1.03
...
Crown    493.9Hz  decay 18.1s  beat 1.74
```

Past a factor of **2** a confirmation appears: beyond that the measured decays
and beats no longer describe the real bowl, and you are listening to an
extrapolation rather than to your instrument.

---

## Parameter checks

A live advisory line reports settings that cannot produce what they claim. The
thresholds come from the mechanism of perception, not from taste.

**Carrier above ~1000 Hz in binaural mode** — the brainstem phase comparison
that creates the beat stops working there, so there is no beat at all. Outside
300–600 Hz it still works, less distinctly (Oster, 1973).

**Beat above 30 Hz in binaural mode** — the two tones stop fusing and are heard
separately. Gamma-range work (40 Hz) uses click trains or flicker, i.e.
**isochronic** delivery. The tool says so and points you to the right mode.

**Sessions under 5 minutes** — published protocols typically run 10–30 minutes,
and effects are measured after several minutes of exposure.

Note that the **Solfeggio** tuning reaches 963 Hz for Crown, which is at the
edge of the usable binaural range, and that the **Gamma** band preset at 40 Hz
is outside it entirely.

---

## What the evidence supports

The entrainment hypothesis — brainwaves synchronising to the beat — **remains
contested**. A 2023 systematic review of 14 studies found 5 supporting it, 8
contradicting it, 1 mixed, and concluded the question cannot be settled yet.

A 2025 randomised study across 16 configurations did find reliable entrainment,
but benefits appeared only for particular combinations of frequency, carrier,
masking noise and timing — wrong settings gave no effect, or reversed it.

Clinical effects are better documented than the mechanism: a 2025 meta-analysis
of 15 randomised trials (over 1000 surgical patients) found reduced anxiety and
post-operative pain versus non-binaural control audio. Whether that works
through entrainment or through something else is not established.

**Chakra and Solfeggio frequencies** are a tuning convention offered here for
composition. They have no measured physiological basis. The presets are
starting points, not protocols.

### What is actually well established

That the auditory cortex follows a rhythmic stimulus is not in doubt — it is the
steady-state response, measured and reproducible. But that is a **sensory
response**, not a change of brain state.

Hippocampal theta, the rhythm these tools invoke by name, is generated by a
dialogue between the medial septum and the hippocampus, riding on resonant
membrane currents and rhythmic inhibition. It is **electrical and local**;
sound is a **mechanical wave** in air. A 6 Hz sound envelope and 6 Hz ionic
currents in the hippocampus share nothing but the number.

If you want theta for real, the levers with evidence behind them are
behavioural: locomotion, memory encoding, and **slow breathing** — around 6
breaths per minute hits the baroreflex resonance and measurably shifts autonomic
balance. A guided meditation that paces breathing acts on a documented
mechanism; the sound bed makes it pleasant and holds attention, which is not
nothing, but is not the same claim.

---

## Empty fields

Every numeric field tolerates being emptied. A `DoubleVar` bound to an Entry
raises as soon as the box is cleared, and each button reading it dies with it —
clearing `Duration` to retype it used to make `+ Add`, `Play`, `Export` and
`Generate audio` all throw. Reads now fall back to a default, write it back into
the field so the value used is visible, and say which field was empty:

```
Duration was empty -- using 300.
```

---

## Requirements

| Package | Why |
|---|---|
| **NumPy** | synthesis | 
| **soundfile** | reading and writing audio |
| **Tkinter** | the interface (usually preinstalled) |
| **pygame** | audio preview only — export works without it |
| **ffmpeg** | MP3 / FLAC / OGG output |

```bash
pip install numpy soundfile pygame
sudo apt install ffmpeg          # optional: non-WAV output
```

**No sound card, no preview.** On a headless machine (a compute server, WSL
without an audio server) the preview cannot open a device and says so —
`Export / Generate audio still work without it`. Generate there, listen
elsewhere.

---

## Files

| File | Role |
|---|---|
| `brainwave_studio.py` | everything: engine, interface, standalone entry point |

It runs both ways: `python brainwave_studio.py` for its own window, or as a tab
inside XTTS Voice Studio, which passes it a frame instead. Window-only calls are
skipped in the second case.

---

## Notes for future work

Three paths render audio: `render()` for a single preview, `build_session()` for
the session preview, and `stream_session()` for export. **Any new option must
be wired into all three.** Four bugs have come from adding one on the preview
path only — `tone_level` ignored on export, peak normalisation cancelling the
beat level in three places, the bowl envelope missing from the streaming
generator, and `level_drone` crashing the export.

Peak normalisation deserves its own warning: normalising a *mix* to peak
silently undoes every level the user set. Only sources may be normalised (each
generator to unit amplitude before its level is applied); the finished mix is
touched only when it would clip, and it says so.
