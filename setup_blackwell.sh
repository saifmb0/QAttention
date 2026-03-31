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
elif [[ "$CUDA_VER" == 13.* ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu130"
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
    "pytest-timeout>=2.1.0" \
    "ninja"        # required by torch cpp_extension's fast build path

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

    # ── FlashAttention-2 ─────────────────────────────────────────────────────
    # Strategy (three-tier):
    #
    #  1. PREBUILT WHEEL (preferred, no nvcc required)
    #     Download directly from the GitHub releases page.
    #     The .so is already compiled; no cpp_extension version check fires.
    #     Wheel naming: flash_attn-{VER}+cu{CUDA_MAJOR}torch{TORCH_MM}cxx11abi{ABI}
    #     We try the matrix of (CUDA major=12|13) x (torch major.minor from installed
    #     down through 2.5) x (cxx11abi TRUE|FALSE).
    #
    #  2. SOURCE BUILD (fallback, requires exact nvcc == torch.version.cuda)
    #     torch cpp_extension does a hard string comparison; even a minor-version
    #     mismatch (12.8 vs 12.9) aborts the build.
    #
    #  3. SKIP with actionable diagnostic.
    if $PYTHON -c "from flash_attn import flash_attn_varlen_func" 2>/dev/null; then
        info "flash_attn already installed and functional — skipping."
    else
        FA_VERSION="2.8.3"
        FA_BASE="https://github.com/Dao-AILab/flash-attention/releases/download/v${FA_VERSION}"

        # Detect Python ABI tag (e.g. cp312)
        PY_TAG=$($PYTHON -c \
            "import sys; v=sys.version_info; print(f'cp{v.major}{v.minor}')" \
            2>/dev/null || echo "cp312")

        # Installed torch major.minor (e.g. "2.11", "2.8")
        TORCH_MM=$($PYTHON -c "
import torch
v = torch.__version__.split('+')[0].split('.')
print(v[0] + '.' + v[1])
" 2>/dev/null || echo "")

        # CUDA major from torch.version.cuda (e.g. "cu12" or "cu13")
        CUDA_MAJ=$($PYTHON -c "
import torch
maj = (torch.version.cuda or '12').split('.')[0]
print('cu' + maj)
" 2>/dev/null || echo "cu12")

        FA_INSTALLED=0

        # ── Tier 1: prebuilt wheels ─────────────────────────────────────────
        info "Trying FlashAttention-2 prebuilt wheels (no nvcc required) …"
        for _cu in "$CUDA_MAJ" "cu12" "cu13"; do
            for _tmm in "$TORCH_MM" "2.8" "2.7" "2.6" "2.5"; do
                [[ -z "$_tmm" ]] && continue
                for _abi in "TRUE" "FALSE"; do
                    _whl="flash_attn-${FA_VERSION}+${_cu}torch${_tmm}cxx11abi${_abi}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
                    _url="${FA_BASE}/${_whl}"
                    # HEAD check first to avoid a full download on 404
                    _http=$(curl -o /dev/null -sILw "%{http_code}" --max-time 15 "$_url" 2>/dev/null || echo "000")
                    if [[ "$_http" == "200" ]]; then
                        info "  Found: ${_whl}  — installing …"
                        if $PIP install --quiet "${_url}" 2>/dev/null; then
                            # Verify the .so actually loads (catches ABI mismatches)
                            if $PYTHON -c "from flash_attn import flash_attn_varlen_func" 2>/dev/null; then
                                FA_INSTALLED=1
                                info "flash_attn installed via prebuilt wheel."
                                break 3
                            else
                                warn "  Wheel loaded but import failed — rolling back."
                                $PIP uninstall -y flash-attn 2>/dev/null || true
                            fi
                        fi
                    fi
                done
            done
        done

        # ── Tier 2: source build (only if nvcc exactly matches torch.version.cuda) ─
        if [[ $FA_INSTALLED -eq 0 ]]; then
            NVCC_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || true)
            TORCH_CUDA_VER=$($PYTHON -c "import torch; print(torch.version.cuda or '')" 2>/dev/null || true)
            if [[ -n "$NVCC_VER" && -n "$TORCH_CUDA_VER" && "$NVCC_VER" == "$TORCH_CUDA_VER" ]]; then
                info "Prebuilt wheel not found — building from source (nvcc ${NVCC_VER} = torch.cuda) …"
                MAX_JOBS=4 $PIP install flash-attn \
                    --no-binary flash-attn \
                    --no-cache-dir \
                    --no-build-isolation \
                    && FA_INSTALLED=1 \
                    && info "flash_attn installed from source." \
                    || warn "flash_attn source build failed."
            fi
        fi

        # ── Tier 3: skip with diagnostic ──────────────────────────────────
        if [[ $FA_INSTALLED -eq 0 ]]; then
            warn "flash_attn not installed — benchmark will show n/a for FA2 baselines."
            warn "  Tried prebuilt wheels: cu={${CUDA_MAJ},cu12,cu13} × torch={${TORCH_MM},2.8,2.7,2.6,2.5}"
            warn "  nvcc=${NVCC_VER:-<not found>}, torch.cuda=${TORCH_CUDA_VER:-<not found>}"
            warn "  Permanent fix: nvcr.io/nvidia/pytorch:25.xx-py3 (nvcc matches torch)."
        fi
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
