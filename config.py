import platform
import subprocess
from pathlib import Path

HOME = Path.home()
MODELS_DIR = HOME / "models"
SESSIONS_DIR = Path(__file__).parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

# Model name substrings used to pick the best available GGUF in MODELS_DIR.
# First match wins.
INTERVIEW_MODEL_PRIORITY = ["8B", "9B", "7B", "4B", "14B", "27B"]
SYNTHESIS_MODEL_PRIORITY = ["14B", "27B", "8B", "9B", "7B", "4B"]

# Set to False to load a separate (larger) model for final prompt synthesis.
# On memory-constrained systems keep True.
USE_SINGLE_MODEL = True

N_CTX = 8192
INTERVIEW_TEMPERATURE = 0.7
SYNTHESIS_TEMPERATURE = 0.2
MAX_TOKENS_INTERVIEW = 1024
MAX_TOKENS_SYNTHESIS = 2048


def get_n_gpu_layers() -> int:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return -1  # Apple Metal
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        return -1  # NVIDIA CUDA
    except Exception:
        return 0  # CPU only


N_GPU_LAYERS = get_n_gpu_layers()
