#!/usr/bin/env bash
# CommandClaw one-click setup
# Works on Linux (NVIDIA/AMD/CPU) and macOS (Apple Silicon / Intel)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
PYTHON=""

# ── find Python 3.10+ ─────────────────────────────────────────────────────────
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(sys.version_info[:2])")
        if "$candidate" -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null; then
            PYTHON=$(command -v "$candidate")
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.10+ not found. Install it first."
    exit 1
fi
echo "Using Python: $PYTHON ($("$PYTHON" --version))"

# ── create venv ───────────────────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment…"
    "$PYTHON" -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip --quiet

# ── install base requirements ─────────────────────────────────────────────────
echo "Installing base dependencies…"
pip install --quiet rich "duckduckgo-search>=6.0" "huggingface_hub>=0.20"

# ── detect GPU backend and install llama-cpp-python ──────────────────────────
OS=$(uname -s)
ARCH=$(uname -m)

install_llama_cuda() {
    echo "Installing llama-cpp-python with CUDA support…"
    CMAKE_ARGS="-DGGML_CUDA=on" \
    pip install llama-cpp-python --no-build-isolation --quiet
}

install_llama_metal() {
    echo "Installing llama-cpp-python with Metal (Apple Silicon) support…"
    CMAKE_ARGS="-DGGML_METAL=on" \
    pip install llama-cpp-python --no-build-isolation --quiet
}

install_llama_cpu() {
    echo "Installing llama-cpp-python (CPU only)…"
    pip install llama-cpp-python --quiet
}

if python -c "import llama_cpp" 2>/dev/null; then
    echo "llama-cpp-python already installed, skipping."
elif [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    install_llama_metal
elif command -v nvidia-smi &>/dev/null; then
    install_llama_cuda
else
    echo "No NVIDIA GPU detected."
    read -r -p "Do you have an AMD GPU (ROCm)? [y/N] " rocm
    if [[ "$rocm" =~ ^[Yy]$ ]]; then
        echo "Installing llama-cpp-python with ROCm/HIP support…"
        CMAKE_ARGS="-DGGML_HIPBLAS=on" \
        pip install llama-cpp-python --no-build-isolation --quiet
    else
        install_llama_cpu
    fi
fi

# ── download models ───────────────────────────────────────────────────────────
echo ""
echo "Setting up models…"
python "$SCRIPT_DIR/download_models.py"

# ── create launcher ───────────────────────────────────────────────────────────
cat > "$SCRIPT_DIR/commandclaw.sh" <<EOF
#!/usr/bin/env bash
SCRIPT_DIR="\$(cd "\$(dirname "\$0")" && pwd)"
source "\$SCRIPT_DIR/.venv/bin/activate"
cd "\$SCRIPT_DIR"
exec python main.py "\$@"
EOF
chmod +x "$SCRIPT_DIR/commandclaw.sh"

echo ""
echo "Setup complete."
echo "Run:  bash $SCRIPT_DIR/commandclaw.sh"
