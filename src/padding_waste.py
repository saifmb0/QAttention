"""
padding_waste.py
================
Padding waste characterisation for speculative-decoding verification batches.

Model
-----
Each request i in a batch carries:
  ctx_i     : context length (prefix of previously verified tokens)
  gamma_i   : number of draft tokens on one path (speculative window)
  tree_depth: depth of draft tree → total draft tokens = num_tree_nodes(b, d)

The full sequence length for request i is:

    L_i = ctx_i + num_tree_nodes(b_i, d_i)

Standard batched attention pads all sequences to L_max = max(L_i).

Waste metrics
-------------
Token-level padding ratio (linear):
    r_token = (B * L_max - sum(L_i)) / (B * L_max)

Attention-compute padding ratio (quadratic, relevant for FLOPs):
    r_attn  = (B * L_max^2 - sum(L_i^2)) / (B * L_max^2)

The quadratic ratio is the primary motivation figure because it directly
corresponds to wasted QK^T multiply-accumulate operations.

Public API
----------
  sequence_lengths(batch_size, ctx_len, branching_factor, depth)
      -> list[int]
  padding_ratios(seq_lens)
      -> dict with keys 'token', 'attn', 'L_max', 'L_mean', 'sum_L', 'sum_L2'
  sweep(batch_sizes, ctx_lens, gammas, depths, branching_factors)
      -> pd.DataFrame
"""

from __future__ import annotations

import itertools
import math
from typing import Sequence

import numpy as np
import pandas as pd

from .tree_mask import num_tree_nodes


# ---------------------------------------------------------------------------
# Sequence-length model
# ---------------------------------------------------------------------------

def sequence_lengths(
    batch_size: int,
    ctx_len: int,
    branching_factor: int,
    depth: int,
    ctx_variance: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """
    Generate per-request sequence lengths for one batch configuration.

    Parameters
    ----------
    batch_size       : B
    ctx_len          : base context length shared by all requests
    branching_factor : b for the draft tree
    depth            : d for the draft tree
    ctx_variance     : if > 0, add uniform noise ±ctx_variance to each
                       context length (simulates heterogeneous prefill)
    rng              : numpy random generator (optional)

    Returns
    -------
    List of B integer sequence lengths.
    """
    draft_tokens = num_tree_nodes(branching_factor, depth)
    if ctx_variance > 0.0 and rng is None:
        rng = np.random.default_rng(42)

    lens: list[int] = []
    for _ in range(batch_size):
        c = ctx_len
        if ctx_variance > 0.0:
            delta = int(rng.integers(-int(ctx_variance), int(ctx_variance) + 1))
            c = max(1, ctx_len + delta)
        lens.append(c + draft_tokens)
    return lens


def padding_ratios(seq_lens: list[int]) -> dict:
    """
    Compute padding waste metrics for a list of per-request sequence lengths.

    Returns a dict:
        L_max   : int
        L_mean  : float
        sum_L   : int
        sum_L2  : int    (sum of squared lengths — proportional to FLOPs)
        token   : float  token-level linear padding ratio ∈ [0, 1)
        attn    : float  attention quadratic padding ratio ∈ [0, 1)
    """
    B = len(seq_lens)
    L_max = max(seq_lens)
    sum_L = sum(seq_lens)
    sum_L2 = sum(l * l for l in seq_lens)
    padded_tokens = B * L_max
    padded_compute = B * L_max * L_max

    return {
        "B": B,
        "L_max": L_max,
        "L_mean": sum_L / B,
        "sum_L": sum_L,
        "sum_L2": sum_L2,
        "token": (padded_tokens - sum_L) / padded_tokens,
        "attn": (padded_compute - sum_L2) / padded_compute,
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep(
    batch_sizes: Sequence[int],
    ctx_lens: Sequence[int],
    gammas: Sequence[int],        # treated as tree depth d for simplicity
    depths: Sequence[int] | None = None,
    branching_factors: Sequence[int] = (2,),
    ctx_variance: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Enumerate all combinations of (batch_size, ctx_len, gamma/depth, b) and
    record padding waste metrics.

    Parameters
    ----------
    batch_sizes       : list of B values
    ctx_lens          : list of baseline context lengths
    gammas            : list of γ values (= tree depth d when depths is None)
    depths            : if provided, used directly as depth d (gammas ignored)
    branching_factors : list of b values
    ctx_variance      : spread of context lengths within a batch
    seed              : RNG seed for ctx_variance sampling

    Returns
    -------
    pd.DataFrame with one row per configuration.
    """
    if depths is None:
        depths = gammas

    rng = np.random.default_rng(seed)
    rows = []

    for B, ctx, d, b in itertools.product(
        batch_sizes, ctx_lens, depths, branching_factors
    ):
        draft_n = num_tree_nodes(b, d)
        seq_lens = sequence_lengths(B, ctx, b, d, ctx_variance, rng)
        metrics = padding_ratios(seq_lens)
        rows.append(
            {
                "batch_size": B,
                "ctx_len": ctx,
                "tree_depth": d,
                "branching_factor": b,
                "draft_tokens": draft_n,
                "L_max": metrics["L_max"],
                "L_mean": round(metrics["L_mean"], 2),
                "token_padding_ratio": round(metrics["token"], 4),
                "attn_padding_ratio": round(metrics["attn"], 4),
            }
        )

    df = pd.DataFrame(rows)
    # Sort for readability
    df = df.sort_values(
        ["batch_size", "tree_depth", "branching_factor"]
    ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = sweep(
        batch_sizes=[1, 4, 8, 16],
        ctx_lens=[128],
        gammas=[1, 2, 3, 4, 5],
        branching_factors=[2, 3],
    )
    print(df.to_string(index=False))
