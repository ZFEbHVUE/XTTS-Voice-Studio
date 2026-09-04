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
**`− Del`** to remove it.

### Keeping and sharing a session

| Button | What it does |
|---|---|
| `📋 Paste (import)` / `📄 Copy (export)` | move a session through the clipboard |
| `💾 Save…` / `📂 Load…` | keep one as a readable `.seg` file |

Pasting or loading into a session that already has segments asks whether to
**add** them after the existing ones or **replace** the list — building a long
session out of shorter saved pieces is the obvious use.

Everything a segment carries is written out, not just the first few fields: a
measured bowl, a rotation, a per-segment drone or a forced band all come back as
they were. Only values that differ from the default are emitted, so a plain
segment stays one readable line.

The list shows only what is actually set. A segment carrying measured bowl modes
says so, because the main panel's carrier and beat are not read at all in that
case:

```
bowl | 6 modes 473Hz+ Delta 1 ramp | 300s | beat -10dB | rot 6.0/min
bina | 400Hz | 6.0Hz | 600s
```

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

Select mode **Bowl**, then **`Edit bowl…`**. One row per vibration mode:

| Column | What it is |
|---|---|
| `freq_hz` | the mode's frequency — the only required field |
| `amp` | its loudness relative to the strongest mode |
| `decay_s` | how long it rings; higher modes die first |
| `beat_hz` | **this mode's own** beat rate |
| `beat_dB` | how strongly it beats. 0 = full, −6 ≈ 74 %, −20 ≈ 36 % of modulation depth |
| `delivery` | `mono` two tones summed · `bina` one per ear · `iso` the tone gated |
| `band` | force the beat onto delta/theta/alpha/beta/gamma instead of the measured value |
| `ramp_hz` | beat target at the end of the segment, 0 = steady |
| `duty` | gate width — **iso only**, greyed otherwise |
| `stereo` | gate offset between the ears — **iso only** |
| `rot/min`, `depth` | this mode's own slow constant-power pan |
| `note → freq` | pick a named frequency (any tuning × chakra, or a Solfeggio) to fill `freq_hz` |

Every row is independent: one partial can beat binaurally in theta while another
pulses isochronically and a third simply rings.

`Save…` and `Load…` write a readable `.bowl` file, so a measured instrument
outlives the segment it was used in. Three starting presets (small / medium /
large), `+ row` to add one, `✕` to remove one, `Check` to see what was
understood before applying, `Clear` to return to the ratio-based bowl.

Short tables still work: a 4-column list — or a bare column of frequencies read
off a phone analyser — parses fine, the later fields taking the values that
reproduce the plain measured behaviour.

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

Modes below **2 %** of the strongest one are dropped: that far down the analyser
is picking up measurement noise rather than the bowl, and those rows arrive with
tell-tale nonsense — a 1700 Hz partial ringing for 30 s (the fallback value when
the decay fit fails), or the same mode found twice a few Hz apart. On a real
recording this took a 16-row table back to the 8 modes that carry the sound.

**For a good recording:** one strike, let it ring to the end, nothing else in
the file. The longer the tail, the more accurate the decay times.

### Ramps restart with each strike

A ramp set in the table (`beat_hz` → `ramp_hz`) completes once per strike, on the
same period as the envelope — not spread over the whole segment.

That distinction cost an evening. `▶ Play` with `Loop` renders only 12 s and
repeats them, so the full ramp was obvious there; the same ramp in a 300 s
segment took five minutes and was inaudible. The preview was promising something
the segment never delivered, which made Play and Add sound like different
instruments. Measured after the fix: identical behaviour in a 12 s and a 300 s
segment, and identical across preview, session and export.

### A sequence of chakras

**`Add all 7`**, beside the chakra buttons, appends one segment per chakra in
order, at the frequencies of the selected tuning. **`Add all 9`** does the same
with the nine Solfeggio frequencies, labelled by frequency because 174 and 285
match no chakra in any convention.

Every other setting on the left is kept, and the panel is put back where you
left it afterwards.

**A sequence, not a sweep.** A bowl does not change pitch while it rings — its
frequency comes from its geometry — so a glissando would sound like a speeded-up
tape rather than like a set of bowls.

#### With a measured bowl loaded

Measured modes override the carrier, so the chakra buttons would otherwise have
no effect. Instead the bowl is **transposed** onto each step: everything scales
by one factor, which is the same as playing bowls of different sizes from the
same workshop.

| | Behaviour |
|---|---|
| Ratios between modes | **unchanged** (verified identical to 0.002) |
| Mode frequencies, measured beats, decays | scale with the factor |
| A beat forced onto a band | **kept** — it was chosen, not measured |
| `beat_dB`, `duty`, `stereo`, rotation | **kept** — choices too |

Past a factor of **2** a confirmation appears: beyond that the measured decays
and beats no longer describe the real bowl.

Note that `Add all 7` gives every segment the same level. For a session where the
level should climb from one chakra to the next, write the `.seg` by hand or
generate it — the file format carries a level per segment.

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
the session preview, and `stream_session()` for export. **Any new option must be
wired into all three.** Bugs from adding one on the preview path only:
`tone_level` ignored on export, peak normalisation cancelling the beat level in
three places, the bowl envelope missing from the streaming generator, and
`level_drone` crashing the export.

**Never leave two definitions of the same function.** Rewriting
`transpose_bowl_modes` and `analyse_bowl_wav` at the top of the file instead of
in place left the stale versions further down, and Python keeps the last one —
so a fix that measured correctly in isolation did nothing in the app. It
happened three times.

**Build the text before opening the file.** `open(path, "w")` truncates at once,
so a failure while formatting leaves a 0-byte file behind. That is exactly how
`Save…` produced empty `.bowl` files while `bowl_modes_to_text` still expected
the old 7-field mode.

**Peak normalisation on a mix silently undoes every level the user set.** Only
sources may be normalised — each generator to unit amplitude before its level is
applied. The finished mix is touched only when it would clip, and it says so.
This is also why the `amp` column of a bowl is relative: the bowl normalises its
own mix, so all four modes at 0.01 sound exactly like all four at 1.0. The
overall level is `Beat level (dB)`, which is applied afterwards and is not
normalised.

**Check the sign of a level formula against a table.** `beat_dB` was inverted:
`gb + (1-gb)·g` instead of `(1-gb) + gb·g`, which made the modulation curve
V-shaped — 100 % at 0 dB, 50 % at −6, and back to 99.7 % at −50. Asking for
almost no beating gave the most. A five-line table of measured values would have
caught it immediately.
