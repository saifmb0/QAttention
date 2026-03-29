#!/usr/bin/env bash
# =============================================================================
# setup_blackwell.sh
# Bootstrap a fresh Ubuntu instance for sd-ragged (blackwell branch)
# Target: NVIDIA RTX PRO 6000 Blackwell Server Edition 94 GB  (SM 12.0, GDDR7)
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
# Detection order:
#  1. nvcc (present in devel images)
#  2. /usr/local/cuda symlink version file (present in runtime images)
#  3. nvidia-smi driver → infer CUDA compat version
#  4. Blackwell-safe default: cu130
CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP "release \K[\d.]+")
if [[ -z "$CUDA_VER" ]]; then
    # Runtime images have a version file instead of nvcc
    CUDA_VER=$(grep -oP 'CUDA Version \K[\d.]+' /usr/local/cuda/version.txt 2>/dev/null \
               || grep -oP '"version"\s*:\s*"\K[0-9.]+' /usr/local/cuda/version.json 2>/dev/null)
fi
if [[ -z "$CUDA_VER" ]]; then
    # Last resort: scan for versioned cuda directories
    CUDA_VER=$(ls -d /usr/local/cuda-* 2>/dev/null \
               | grep -oP 'cuda-\K[\d.]+' | sort -V | tail -1)
fi
[[ -z "$CUDA_VER" ]] && CUDA_VER="none"
info "CUDA version: $CUDA_VER"

# Pick the right torch index URL
if [[ "$CUDA_VER" == 13.* ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu130"
elif [[ "$CUDA_VER" == 12.* ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
elif [[ "$CUDA_VER" == 11.* ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cu130"
    warn "Could not detect CUDA version; defaulting to cu130 (Blackwell SM 12.0)."
fi

info "Installing core requirements from: $TORCH_INDEX"
$PIP install --quiet \
    "torch>=2.2.0" \
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

    # ── Auto-detect CUDA_HOME (required by flash-attn build system) ──────────
    if [[ -z "${CUDA_HOME:-}" ]]; then
        for _cand in /usr/local/cuda /usr/cuda $(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1); do
            if [[ -f "${_cand}/bin/nvcc" ]]; then
                export CUDA_HOME="$_cand"
                export PATH="${CUDA_HOME}/bin:${PATH}"
                info "Auto-detected CUDA_HOME=${CUDA_HOME}"
                break
            fi
        done
    fi
    if [[ -z "${CUDA_HOME:-}" ]]; then
        warn "CUDA_HOME not set and nvcc not found — flash-attn build will be skipped."
        warn "Use a 'devel' Docker image (e.g. pytorch/pytorch:*-devel) to get nvcc."
    fi

    # FlashAttention-2
    if $PYTHON -c "import flash_attn" 2>/dev/null; then
        info "flash_attn already installed — skipping."
    elif [[ -z "${CUDA_HOME:-}" ]]; then
        warn "flash_attn skipped — nvcc / CUDA_HOME not available."
    else
        info "Building FlashAttention-2 (may take 5–10 min) …"
        $PIP install --quiet flash-attn --no-build-isolation \
            && info "flash_attn installed." \
            || warn "flash_attn build failed — benchmark will skip FA2 comparison."
    fi

    # FlashInfer — try multiple CUDA/torch index URLs
    if $PYTHON -c "import flashinfer" 2>/dev/null; then
        info "flashinfer already installed — skipping."
    else
        info "Installing FlashInfer (trying multiple CUDA index URLs) …"
        CUDA_SHORT=$($PYTHON -c "
import torch; v=torch.version.cuda or '121'
parts=v.split('.')[:2]; print('cu'+''.join(parts))
" 2>/dev/null || echo "cu121")
        TORCH_SHORT=$($PYTHON -c "
import torch; v=torch.__version__
print('torch'+v[:3].replace('.',''))
" 2>/dev/null || echo "torch23")
        FI_INSTALLED=0
        for _fi_url in \
            "https://flashinfer.ai/whl/${CUDA_SHORT}/${TORCH_SHORT}/" \
            "https://flashinfer.ai/whl/cu124/torch26/" \
            "https://flashinfer.ai/whl/cu124/torch25/" \
            "https://flashinfer.ai/whl/cu121/torch23/"; do
            if $PIP install --quiet flashinfer -i "${_fi_url}" 2>/dev/null; then
                FI_INSTALLED=1
                info "flashinfer installed from ${_fi_url}"
                break
            fi
        done
        if [[ $FI_INSTALLED -eq 0 ]]; then
            warn "flashinfer install failed (no pre-built wheel for ${CUDA_SHORT}/${TORCH_SHORT}) — benchmark will skip FlashInfer comparison."
        fi
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
    sm = (p.major, p.minor)
    if sm >= (12, 0):
        arch, tier = 'Blackwell', 'SM120 configs'
    elif sm >= (8, 9):
        arch, tier = 'Lovelace', 'SM89 configs'
    elif sm >= (8, 0):
        arch, tier = 'Ampere', 'SM75 configs'
    elif sm >= (7, 5):
        arch, tier = 'Turing', 'SM75 configs'
    else:
        arch, tier = f'SM{p.major}{p.minor}', 'SM75 configs'
    print(f"  Architecture : {arch}  ({tier} active)")
EOF

info "Setup complete!  Run:  bash run_blackwell.sh"
