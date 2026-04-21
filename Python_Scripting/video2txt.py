#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video2txt.py — Extract audio from a video file then transcribe to XTTS-ready text.

Wrapper around transcribeSong2txt_with_pause.py:
  1. ffmpeg extracts audio to a temporary WAV file
  2. transcribeSong2txt_with_pause.py transcribes it (faster-whisper)
  3. Temporary WAV is deleted

Usage:
  python video2txt.py input.mp4 output.txt --model medium --lang fr --pause 0.3 --device cuda
  python video2txt.py input.mp4 output.txt --model large-v3 --lang fr --pause 0.1 --device cuda --pitch --vad

Arguments:
  input         Video file (.mp4 .mkv .avi .mov .flv .webm .wmv .m4v .ts .mpg)
  output        Output text file (.txt)
  --model       Whisper model: tiny / base / small / medium / large-v3 / turbo  (default: small)
  --lang        Language code: fr / en / es / de / ...                           (default: fr)
  --pause       Minimum silence duration in seconds to insert a pause marker     (default: 0.3)
  --device      cpu or cuda                                                       (default: cpu)
  --pitch       Add per-word pitch annotation [p:+2] etc.
  --vad         Enable Silero VAD pre-filtering (skips silence before Whisper)
"""

import argparse
import os
import sys
import subprocess
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRIBE = os.path.join(SCRIPT_DIR, 'transcribeSong2txt_with_pause.py')


def extract_audio(video_path, wav_path):
    """Extract audio track from video to a mono 16kHz WAV via ffmpeg."""
    print(f"[*] Extracting audio from: {os.path.basename(video_path)}")
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vn',                    # no video
        '-ac', '1',               # mono
        '-ar', '16000',           # 16 kHz (optimal for Whisper)
        '-acodec', 'pcm_s16le',   # standard WAV
        wav_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[!] ffmpeg error:\n{result.stderr[-500:]}")
            return False
        print(f"[OK] Audio extracted -> {os.path.basename(wav_path)}")
        return True
    except FileNotFoundError:
        print("[!] ffmpeg not found. Install it with: sudo apt install ffmpeg")
        return False


_proc_holder = [None]

def main():
    parser = argparse.ArgumentParser(
        description='Extract audio from video then transcribe to XTTS-ready text.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('input',         help='Input video file')
    parser.add_argument('output',        help='Output text file')
    parser.add_argument('--model',       default='small',
                        choices=['tiny','base','small','medium','large-v2','large-v3','turbo'],
                        help='Whisper model (default: small)')
    parser.add_argument('--lang',        default='fr',   help='Language code (default: fr)')
    parser.add_argument('--pause',       default='0.3',  help='Min silence for pause marker in s (default: 0.3)')
    parser.add_argument('--device',      default='cuda', choices=['cpu','cuda'],
                        help='Device (default: cpu)')
    parser.add_argument('--pitch',       action='store_true', help='Add per-word pitch annotation')
    parser.add_argument('--vad',         action='store_true', help='Enable Silero VAD pre-filtering')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[!] Input file not found: {args.input}")
        sys.exit(1)

    if not os.path.exists(TRANSCRIBE):
        print(f"[!] transcribeSong2txt_with_pause.py not found at: {TRANSCRIBE}")
        sys.exit(1)

    # Extract audio to temp WAV
    tmp = tempfile.NamedTemporaryFile(suffix='_video2txt.wav', delete=False)
    tmp.close()

    try:
        if not extract_audio(args.input, tmp.name):
            sys.exit(1)

        # Build transcription command
        # transcribeSong2txt_with_pause.py uses positional args: audio output model pause lang
        cmd = [
            sys.executable, TRANSCRIBE,
            tmp.name,
            args.output,
            args.model,
            args.pause,
            args.lang,
            '--device', args.device
        ]
        if args.pitch: cmd.append('--pitch')
        if args.vad:   cmd.append('--vad')

        print(f"[*] Transcribing ({args.model}, {args.lang.upper()}, device={args.device})...")

        # Use Popen so the process can be killed by the GUI Stop button.
        # os.setsid() creates a new process group — when the GUI terminates
        # video2txt.py, SIGTERM propagates to ffmpeg/whisper children too.
        import signal
        proc = subprocess.Popen(cmd, start_new_session=True)
        _proc_holder[0] = proc

        def _kill_tree():
            """Kill the child process and its entire process group."""
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                import time; time.sleep(0.5)
                try:
                    os.killpg(pgid, _sig.SIGKILL)  # force-kill if still alive
                except ProcessLookupError:
                    pass
            except Exception:
                try: proc.kill()
                except Exception: pass

        def _on_sigterm(sig, frame):
            _kill_tree()
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
            os._exit(1)

        def _on_sigint(sig, frame):
            _kill_tree()
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
            os._exit(1)

        signal.signal(signal.SIGTERM, _on_sigterm)
        signal.signal(signal.SIGINT,  _on_sigint)

        try:
            proc.wait()
        except KeyboardInterrupt:
            _kill_tree()
        sys.exit(proc.returncode if proc.returncode is not None else 1)

    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
            print(f"[*] Temp audio removed.")


if __name__ == '__main__':
    main()
