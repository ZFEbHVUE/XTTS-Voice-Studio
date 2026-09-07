#!/usr/bin/env bash
# XTTS Voice Studio — one-shot installer for Linux (and WSL).
#
# Creates the conda environment, installs the pinned dependencies, checks the
# system tools that pip cannot provide, and puts a launcher in the applications
# menu and on the desktop.
#
#   ./install.sh              install into an env called "xtts"
#   ./install.sh --name myenv use another environment name
#   ./install.sh --cpu        install the CPU-only build of torch
#   ./install.sh --no-icon    skip the desktop launcher
#
# It is safe to run twice: an existing environment is reused, not rebuilt.

set -uo pipefail

ENV_NAME="xtts"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
MAKE_ICON=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [ $# -gt 0 ]; do
    case "$1" in
        --name)    ENV_NAME="${2:?--name needs a value}"; shift 2 ;;
        --cpu)     TORCH_INDEX=""; shift ;;
        --no-icon) MAKE_ICON=0; shift ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *)         echo "Unknown option: $1"; exit 1 ;;
    esac
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m[!] %s\033[0m\n' "$*"; }
die()  { printf '\033[31m[x] %s\033[0m\n' "$*"; exit 1; }

# ── conda ────────────────────────────────────────────────────────────────────
say "Looking for conda"
if ! command -v conda >/dev/null 2>&1; then
    die "conda not found. Install Miniconda first:
    https://docs.conda.io/en/latest/miniconda.html
Then open a new shell and run this script again."
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
echo "    $(conda --version), base at $(conda info --base)"

# ── environment ──────────────────────────────────────────────────────────────
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    say "Environment '$ENV_NAME' already exists — reusing it"
else
    say "Creating environment '$ENV_NAME' (Python 3.10)"
    conda create -y -n "$ENV_NAME" python=3.10 || die "could not create the environment"
fi
conda activate "$ENV_NAME" || die "could not activate '$ENV_NAME'"
echo "    $(python --version), $(which python)"

# ── torch first, matched to CUDA ─────────────────────────────────────────────
# Installed before the rest so pip does not pull a different build as a
# dependency of something else.
say "Installing PyTorch"
if [ -n "$TORCH_INDEX" ]; then
    echo "    CUDA 12.1 build (use --cpu for the CPU-only one)"
    pip install --quiet "torch==2.5.1" "torchaudio==2.5.1" --index-url "$TORCH_INDEX" \
        || die "torch install failed"
else
    echo "    CPU-only build"
    pip install --quiet "torch==2.5.1" "torchaudio==2.5.1" || die "torch install failed"
fi

# ── everything else ──────────────────────────────────────────────────────────
say "Installing the Python dependencies"
[ -f "$HERE/requirements.txt" ] || die "requirements.txt not found next to this script"
pip install --quiet -r "$HERE/requirements.txt" || die "dependency install failed"

# XTTS asks for this on first run and blocks waiting for an answer otherwise.
conda env config vars set COQUI_TOS_AGREED=1 -n "$ENV_NAME" >/dev/null 2>&1 || true

# ── system tools pip cannot provide ──────────────────────────────────────────
say "Checking the tools that are not Python packages"
MISSING=()
for tool in ffmpeg rubberband; do
    if command -v "$tool" >/dev/null 2>&1; then
        echo "    $tool: found"
    else
        echo "    $tool: MISSING"
        MISSING+=("$tool")
    fi
done
python -c "import tkinter" >/dev/null 2>&1 \
    && echo "    tkinter: found" \
    || { echo "    tkinter: MISSING"; MISSING+=("python3-tk"); }

if [ ${#MISSING[@]} -gt 0 ]; then
    warn "Install the missing ones with your package manager, for example:"
    echo "    sudo apt install ${MISSING[*]/rubberband/rubberband-cli}"
    echo "    (ffmpeg is needed for MP3/FLAC/OGG and video; rubberband for the"
    echo "     age shift and tempo change; tkinter for the interface itself)"
fi

# ── does it import? ──────────────────────────────────────────────────────────
say "Checking the install"
python - <<'PY' || die "the environment does not import cleanly — see the error above"
import importlib, sys
bad = []
for mod in ("numpy", "scipy", "librosa", "soundfile", "torch", "TTS",
            "transformers", "parselmouth", "speechbrain", "sklearn"):
    try:
        importlib.import_module(mod)
    except Exception as e:
        bad.append(f"{mod}: {e}")
if bad:
    print("\n".join(bad)); sys.exit(1)
import torch
print(f"    torch {torch.__version__}, CUDA "
      f"{'available: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available (CPU only)'}")
try:
    import pygame  # noqa: F401
    print("    pygame present — audio preview available")
except Exception:
    print("    pygame missing — export works, preview does not")
PY

# ── launcher ─────────────────────────────────────────────────────────────────
if [ "$MAKE_ICON" -eq 1 ]; then
    say "Creating the launcher"
    CONDA_BASE="$(conda info --base)"
    BIN="$HOME/.local/bin/xtts-studio"
    mkdir -p "$HOME/.local/bin"
    cat > "$BIN" <<EOF
#!/usr/bin/env bash
# Generated by install.sh — starts XTTS Voice Studio in its environment.
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$HERE"
exec python Python_Scripting/xtts_studio.py "\$@"
EOF
    chmod +x "$BIN"

    ICON="$HERE/docs/icon.png"
    [ -f "$ICON" ] || ICON="applications-multimedia"
    DESKTOP="$HOME/.local/share/applications/xtts-studio.desktop"
    mkdir -p "$(dirname "$DESKTOP")"
    cat > "$DESKTOP" <<EOF
[Desktop Entry]
Type=Application
Name=XTTS Voice Studio
Comment=Voice cloning, guided meditation and brainwave audio
Exec=$BIN
Icon=$ICON
Terminal=false
Categories=AudioVideo;Audio;
EOF
    chmod +x "$DESKTOP"
    update-desktop-database "$(dirname "$DESKTOP")" >/dev/null 2>&1 || true

    # A copy on the desktop too, when there is one.
    for d in "$HOME/Desktop" "$HOME/Bureau"; do
        if [ -d "$d" ]; then
            cp "$DESKTOP" "$d/" && chmod +x "$d/xtts-studio.desktop"
            gio set "$d/xtts-studio.desktop" metadata::trusted true >/dev/null 2>&1 || true
            echo "    also on $d"
        fi
    done
    echo "    menu entry: $DESKTOP"
    echo "    command:    xtts-studio  (if ~/.local/bin is on your PATH)"
fi

say "Done"
cat <<EOF
    Start it from the applications menu, from the desktop icon, or with:

        conda activate $ENV_NAME
        python Python_Scripting/xtts_studio.py

    First run downloads the XTTS model (about 2 GB) — that is normal.

    Under WSL, sound and window come from WSLg. If either fails, check that
    ~/.bashrc does not still export PULSE_SERVER or DISPLAY as an IP address:

        export PULSE_SERVER=unix:/mnt/wslg/PulseServer
        export DISPLAY=:0
EOF
