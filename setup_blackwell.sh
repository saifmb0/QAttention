#!/usr/bin/env bash
# =============================================================================
# setup_blackwell.sh
# Bootstrap a fresh Ubuntu instance for sd-ragged (blackwell branch)
# Target: NVIDIA RTX 6000 ADA PRO 96 GB  (SM 8.9 Ada Lovelace, GDDR7)
#
# Usage:
#   bash setup_blackwell.sh              # full setup
#   bash setup_blackwell.sh --no-sota    # skip optional SOTA libs
#   bash setup_blackwell.sh --venv       # use a virtualenv (default: system pip)
# =============================================================================
set -euo pipefail

# ─── Colour helpers ───────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[setup]${NC} $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC}  $*"; }
die()     { echo -e "${RED}[error]${NC} $*"; exit 1; }

# ─── Args ────────────────────────────────────────────────────────────────────
INSTALL_SOTA=1
USE_VENV=0
for arg in "$@"; do
    case $arg in
        --no-sota)   INSTALL_SOTA=0 ;;
        --venv)      USE_VENV=1     ;;
    esac
done

# ─── 0. GPU check ─────────────────────────────────────────────────────────────
info "Checking GPU …"
if ! command -v nvidia-smi &>/dev/null; then
    warn "nvidia-smi not found — skipping GPU check (no CUDA driver or CPU-only CI)"
else
    nvidia-smi --query-gpu=name,driver_version,memory.total \
                --format=csv,noheader
    SM=$(python3 -c "
import subprocess, re
out = subprocess.check_output(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader']).decode()
print(out.strip().replace('.',''))
" 2>/dev/null || echo "unknown")
    info "SM: $SM"
    if [[ "$SM" < "89" && "$SM" != "unknown" ]]; then
        warn "SM $SM detected — this branch is optimised for SM 8.9 (Ada)."
        warn "The kernel will fall back to SM75 autotune configs automatically."
    fi
fi

# ─── 1. System packages ───────────────────────────────────────────────────────
info "Updating system packages …"
if command -v apt-get &>/dev/null; then
    apt-get update -qq
    apt-get install -y -qq git curl wget build-essential python3-pip python3-venv
fi

# ─── 2. Python environment ────────────────────────────────────────────────────
PYTHON="python3"
PIP="pip3"

if [[ $USE_VENV -eq 1 ]]; then
    info "Creating virtual environment …"
    $PYTHON -m venv .venv
    source .venv/bin/activate
    PYTHON="python"
    PIP="pip"
fi

$PYTHON --version
$PIP --version

# ─── 3. CUDA check + PyTorch ──────────────────────────────────────────────────
CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP "release \K[\d.]+" || echo "none")
info "CUDA version: $CUDA_VER"

# Pick the right torch index URL
if [[ "$CUDA_VER" == 12.* ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
elif [[ "$CUDA_VER" == 11.* ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    warn "Could not detect CUDA version; defaulting to cu121 PyTorch wheel."
fi

info "Installing core requirements from: $TORCH_INDEX"
$PIP install --quiet \
    "torch>=2.2.0,<3.0.0" \
    "triton>=2.3.0,<4.0.0" \
    --extra-index-url "$TORCH_INDEX"

$PIP install --quiet \
    "numpy>=1.24.0" \
    "pandas>=2.0.0" \
    "matplotlib>=3.7.0" \
    "pytest>=7.4.0" \
    "pytest-timeout>=2.1.0"

info "Core requirements installed."

# ─── 4. Optional SOTA libraries ───────────────────────────────────────────────
if [[ $INSTALL_SOTA -eq 1 ]]; then
    info "Installing optional SOTA benchmark libraries …"

    # FlashAttention-2
    if $PYTHON -c "import flash_attn" 2>/dev/null; then
        info "flash_attn already installed — skipping."
    else
        info "Building FlashAttention-2 (may take 5–10 min) …"
        $PIP install --quiet flash-attn --no-build-isolation \
            && info "flash_attn installed." \
            || warn "flash_attn build failed — benchmark will skip FA2 comparison."
    fi

    # FlashInfer
    if $PYTHON -c "import flashinfer" 2>/dev/null; then
        info "flashinfer already installed — skipping."
    else
        info "Installing FlashInfer …"
        TORCH_SHORT=$(python3 -c "import torch; v=torch.__version__; print('torch'+v[:3].replace('.',''))" 2>/dev/null || echo "torch23")
        $PIP install --quiet flashinfer \
            -i "https://flashinfer.ai/whl/cu121/${TORCH_SHORT}/" \
            && info "flashinfer installed." \
            || warn "flashinfer install failed — benchmark will skip FlashInfer comparison."
    fi

    # xformers
    if $PYTHON -c "import xformers" 2>/dev/null; then
        info "xformers already installed — skipping."
    else
        info "Installing xformers …"
        $PIP install --quiet xformers \
            && info "xformers installed." \
            || warn "xformers install failed — benchmark will skip xformers comparison."
    fi
else
    info "Skipping optional SOTA libraries (--no-sota)."
fi

# ─── 5. Verify core imports ───────────────────────────────────────────────────
info "Verifying imports …"
$PYTHON - <<'EOF'
import torch, triton, numpy, pandas, matplotlib
print(f"  torch   {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
print(f"  triton  {triton.__version__}")
print(f"  numpy   {numpy.__version__}")
print(f"  pandas  {pandas.__version__}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU     {p.name}  SM{p.major}{p.minor}  {p.total_memory//1024**3} GB")
    ada = (p.major, p.minor) >= (8, 9)
    print(f"  Ada Lovelace (SM 8.9): {'YES' if ada else 'NO (SM75 configs will be used)'}")
EOF

info "Setup complete!  Run:  bash run_blackwell.sh"
