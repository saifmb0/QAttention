#!/usr/bin/env bash
# =============================================================================
# setup.sh
# Bootstrap for sd-ragged
# Target: NVIDIA RTX 4000 Ada (SM 8.9, Lovelace) running CUDA 12.x
# Also works on SM 9.0 (Hopper) and SM 12.0 (Blackwell) — kernel autotuner
# selects the appropriate config tier at runtime.
#
# What this script does:
#   1. Installs PyTorch 2.8.0 + CUDA 12.1 — stable, reproducible, known-good
#      wheel that matches FlashInfer's pre-built index.
#   2. Does NOT install flash-attn.
#      Reason: every torch/CUDA minor-version bump breaks the .so ABI, and
#      torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION) uses the same
#      FlashAttention-2 kernel natively without the separate package.
#   3. Installs FlashInfer (flashinfer-python + flashinfer-cubin +
#      flashinfer-jit-cache) from the cu129 index.  The cubin package ships
#      pre-compiled SM90 Hopper kernels — no JIT / nvcc build at runtime.
#      FlashInfer is used as the Naive baseline ragged-tree-attention competitor.
#
# Usage:
#   bash setup_blackwell.sh              # full setup
#   bash setup_blackwell.sh --no-sota    # skip optional SOTA libs
#   bash setup_blackwell.sh --venv       # isolate in a virtualenv
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${GREEN}[setup]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC}  $*"; }
die()  { echo -e "${RED}[error]${NC} $*"; exit 1; }

INSTALL_SOTA=1
USE_VENV=0
for arg in "$@"; do
    case $arg in
        --no-sota) INSTALL_SOTA=0 ;;
        --venv)    USE_VENV=1     ;;
    esac
done

# ── 0. GPU check ──────────────────────────────────────────────────────────────
info "Checking GPU …"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    SM=$(python3 -c "
import subprocess, re
out = subprocess.check_output(
    ['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader']).decode()
print(out.strip().replace('.',''))
" 2>/dev/null || echo "unknown")
    info "SM: $SM"
    if [[ "$SM" != "unknown" ]]; then
        if [[ "$SM" -lt "75" ]] 2>/dev/null; then
            warn "SM $SM detected — kernel targets SM75+; older hardware may be slow."
        fi
    fi
else
    warn "nvidia-smi not found — GPU check skipped (CPU-only CI or missing driver)."
fi

# ── 1. System packages ────────────────────────────────────────────────────────
info "Updating system packages …"
if command -v apt-get &>/dev/null; then
    sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git curl wget build-essential python3-pip python3-venv
fi

# ── 2. Python environment ─────────────────────────────────────────────────────
PYTHON="python3"
PIP="pip3"

if [[ $USE_VENV -eq 1 ]]; then
    info "Creating virtual environment …"
    $PYTHON -m venv .venv
    source .venv/bin/activate
    PYTHON="python"; PIP="pip"
fi

$PYTHON --version
$PIP --version

# ── 3. Core: PyTorch 2.8.0 + CUDA 12.1 ───────────────────────────────────────
# We pin to 2.8.0 because:
#   • FlashInfer pre-built wheels exist for exactly cu121/torch2.8
#   • 2.8.0 is stable and has matured FA-2 SDPA + bf16 support on SM90
#   • PyTorch 2.9+ wheels were not yet reliably packaged for cu121 at branch time
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
info "Installing PyTorch 2.8.0 (cu121) …"
$PIP install --quiet \
    "torch==2.8.0" \
    "triton>=3.0.0,<4.0.0" \
    --extra-index-url "$TORCH_INDEX"

$PIP install --quiet \
    "numpy>=1.24.0" \
    "pandas>=2.0.0" \
    "matplotlib>=3.7.0" \
    "pytest>=7.4.0" \
    "pytest-timeout>=2.1.0" \
    "ninja"

info "Core requirements installed."

# ── 4. Optional SOTA libraries ────────────────────────────────────────────────
if [[ $INSTALL_SOTA -eq 1 ]]; then
    info "Installing optional SOTA benchmark libraries …"

    # NOTE: flash-attn is intentionally NOT installed.
    # PyTorch SDPA (enable_flash=True) provides identical kernel coverage for
    # our benchmarking purposes without the fragile .so ABI binding.
    info "  flash-attn: skipped (using torch.nn.functional.scaled_dot_product_attention)"

    # ── FlashInfer ────────────────────────────────────────────────────────────
    # Three packages for full Hopper support (pre-compiled cubins, no JIT build):
    #   flashinfer-python      — core Python API + JIT fallback
    #   flashinfer-cubin        — pre-compiled .cubin files (SM90 Hopper)
    #   flashinfer-jit-cache    — cached JIT artefacts for common configs
    # Index: https://flashinfer.ai/whl/cu129
    #
    # With cubins installed the SM90 kernels load directly — no ninja/nvcc
    # build at first import, and no curand_kernel.h dependency.
    FLASHINFER_INDEX="https://flashinfer.ai/whl/cu129"
    if $PYTHON -c "import flashinfer" 2>/dev/null; then
        info "  flashinfer: already installed — skipping."
    else
        info "  Installing FlashInfer (cubins + JIT cache, cu129) …"
        if $PIP install --quiet \
               "flashinfer-python" "flashinfer-cubin" "flashinfer-jit-cache" \
               --extra-index-url "$FLASHINFER_INDEX" \
               2>/dev/null \
           && $PYTHON -c "import flashinfer; print('flashinfer', getattr(flashinfer, '__version__', '?'))" 2>/dev/null; then
            info "  flashinfer (cubins) installed successfully."
        else
            warn "  flashinfer install failed — FlashInfer baselines will be n/a."
            warn "  Manual: pip install flashinfer-python flashinfer-cubin flashinfer-jit-cache \\"
            warn "    --extra-index-url https://flashinfer.ai/whl/cu129"
        fi
    fi

    # ── DeFT (arXiv:2404.00242, ICLR'25) ─────────────────────────────────────
    # LINs-lab/DeFT is public but requires torch==2.5.1 + SGLang — incompatible
    # with our torch 2.8.0 environment.
    #
    # PanZaifeng/FastTree-Artifact/kernel_bench/ contains a standalone Triton
    # implementation (DeFT.py + kv_tree_simple.py) with only triton+numpy deps.
    # We sparse-clone just that subdirectory.
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    DEFT_KERNEL_DIR="${REPO_ROOT}/third_party/FastTree/kernel_bench"
    if [[ -f "${DEFT_KERNEL_DIR}/DeFT.py" ]]; then
        info "  DeFT (FastTree Triton kernel): already present — skipping clone."
    else
        info "  Cloning FastTree-Artifact/kernel_bench for DeFT Triton kernel …"
        _FT_PARENT="${REPO_ROOT}/third_party/FastTree"
        mkdir -p "${_FT_PARENT}"
        if git clone --depth=1 --filter=blob:none --sparse \
               https://github.com/PanZaifeng/FastTree-Artifact.git \
               "${_FT_PARENT}" 2>/dev/null \
           && (cd "${_FT_PARENT}" && git sparse-checkout set kernel_bench 2>/dev/null) \
           && [[ -f "${DEFT_KERNEL_DIR}/DeFT.py" ]]; then
            info "  DeFT kernel available: ${DEFT_KERNEL_DIR}/DeFT.py"
        else
            warn "  FastTree clone failed — DeFT baseline will be n/a."
            warn "  Manual: git clone --depth=1 --filter=blob:none --sparse \\"
            warn "    https://github.com/PanZaifeng/FastTree-Artifact.git third_party/FastTree"
            warn "  Then: cd third_party/FastTree && git sparse-checkout set kernel_bench"
        fi
    fi

    # ── EAGLE (NeurIPS'25) — speculative decoding framework ──────────────────
    # SafeAILab/EAGLE provides EaModel with eagenerate() for Eagle-1/2/3.
    # Used in e2e_benchmark.py for real speculative decoding evaluation.
    #
    # eagle-llm==3.0.0 pins torch==2.0.1 in its metadata, which conflicts with
    # our torch 2.8+.  We install with --no-deps to skip that metadata check —
    # EAGLE's runtime code is compatible with any recent torch.
    # We install transformers, accelerate, fschat, and sentencepiece separately
    # so EAGLE's actual runtime imports are satisfied.
    #
    # transformers must be up-to-date: EAGLE's modeling_qwen3_kv.py uses
    # LossKwargs, auto_docstring, can_return_tuple which were added in recent
    # releases (4.47+ / 4.50+).  We unconditionally upgrade to latest to avoid
    # a stale install blocking the import — even if EAGLE is already present.
    # Pin to EAGLE's tested version range.  EAGLE's requirements.txt specifies
    # transformers>=4.53.1, and that is the version the EAGLE code was actually
    # written and tested against.  Using a newer (or much older) version can
    # silently corrupt generation (wrong RoPE, wrong dtype, wrong output dict
    # keys) without raising any errors.  We pin <5.0 to avoid major-version
    # breaks; remove the upper bound once EAGLE explicitly supports 5.x.
    info "  Installing transformers==4.53.1 (EAGLE-required version) …"
    $PIP install --quiet "transformers==4.53.1" 2>/dev/null || \
        $PIP install --quiet "transformers>=4.53.1,<5.0" 2>/dev/null || \
        warn "  transformers install failed — EAGLE may not import correctly."

    if $PYTHON -c "from eagle.model.ea_model import EaModel" 2>/dev/null; then
        info "  EAGLE: already installed and importable."
    else
        info "  Installing EAGLE (speculative decoding framework) …"
        # Install remaining runtime deps that --no-deps skips.
        # accelerate: EAGLE was authored for 0.26.0; 1.x changed device-map hooks
        # significantly.  Pin to 0.x to avoid silent corruption with device_map="auto".
        $PIP install --quiet \
            "accelerate>=0.26.0,<1.0" \
            "sentencepiece" \
            "fschat" \
            2>/dev/null || true
        # Install EAGLE itself — bypass the erroneous torch==2.0.1 metadata pin
        if $PIP install --quiet --no-deps \
               "git+https://github.com/SafeAILab/EAGLE.git" \
               2>/dev/null \
           && $PYTHON -c "from eagle.model.ea_model import EaModel; print('EAGLE installed OK')" 2>/dev/null; then
            info "  EAGLE installed successfully."
        else
            warn "  EAGLE install failed — E2E benchmark will require --skip-generation."
            warn "  Manual: pip install --no-deps git+https://github.com/SafeAILab/EAGLE.git"
            warn "          pip install --upgrade transformers accelerate sentencepiece fschat"
        fi
    fi

else
    info "Skipping optional SOTA libraries (--no-sota)."
fi

# ── 5. Verify imports ─────────────────────────────────────────────────────────
info "Verifying imports …"
$PYTHON - <<'EOF'
import torch, triton, numpy, pandas, matplotlib
print(f"  torch   {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
print(f"  triton  {triton.__version__}")
print(f"  numpy   {numpy.__version__}")
print(f"  pandas  {pandas.__version__}")

if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    vram = p.total_memory // 1024**3
    sm   = (p.major, p.minor)
    if sm >= (12, 0):
        arch, tier = "Blackwell", "SM120 configs (use blackwell branch)"
    elif sm >= (9, 0):
        arch, tier = "Hopper", "SM90 configs active"
    elif sm >= (8, 9):
        arch, tier = "Lovelace", "SM89 configs active"
    elif sm >= (8, 0):
        arch, tier = "Ampere", "SM75 configs active"
    elif sm >= (7, 5):
        arch, tier = "Turing", "SM75 configs active"
    else:
        arch, tier = f"SM{p.major}{p.minor}", "SM75 configs active (fallback)"
    print(f"  GPU     {p.name}  SM{p.major}{p.minor}  {vram} GB")
    print(f"  Arch    {arch} — {tier}")

# Optional libs
for lib, label in [("flashinfer", "FlashInfer"), ("xformers", "xformers")]:
    try:
        m = __import__(lib)
        ver = getattr(m, "__version__", "?")
        print(f"  {label:<12} {ver}  OK")
    except Exception:
        print(f"  {label:<12} NOT installed (optional)")
EOF

info "Setup complete!  Run:  bash run_benchmarks.sh"
