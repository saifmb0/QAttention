"""
tree_mask.py
============
Tree-structured causal attention mask construction for draft trees in
speculative decoding (e.g. EAGLE-2).

Terminology
-----------
  branching_factor b : each non-leaf node grows b children
  depth d            : number of edges from root to leaves (root is depth 0)
  num_nodes N        : total nodes in the tree,
                         N = (b^(d+1) - 1) / (b - 1)  for b > 1
                         N = d + 1                      for b = 1

Node numbering: BFS (level-order). Root = 0.
  Level 0  : [0]
  Level 1  : [1 .. b]
  Level 2  : [b+1 .. b+b^2]
  ...

Attention rule for verification pass
--------------------------------------
During speculative decoding the target model processes:
  [context tokens]  +  [draft tokens]

Only the draft prefix is "tree-structured". The context tokens form a linear
prefix that every draft token can attend to. Inside the draft tree, token i
may attend to token j **only if j is an ancestor of i (or j == i)**.

This module generates:
  - parent_array    list[int] of length N  (parent_array[0] = -1 for root)
  - attention_mask  bool ndarray [N, N]   where mask[i, j] = True iff i may
                    attend to j (i.e. j is an ancestor of i or j == i)

The mask is then combined with the context prefix via full-True rows/columns
for context positions.

Public API
----------
  build_tree(b, d)             -> parent_array: list[int]
  ancestors(node, parent_array) -> set[int]
  tree_attention_mask(b, d)    -> np.ndarray[bool, (N, N)]
  full_sequence_mask(ctx_len, b, d) -> np.ndarray[bool, (ctx_len+N, ctx_len+N)]
  verify_tree_mask(b, d)       -> bool  (prints diagnostics)
  num_tree_nodes(b, d)         -> int
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def num_tree_nodes(branching_factor: int, depth: int) -> int:
    """Total number of nodes in a complete b-ary tree of given depth."""
    b, d = branching_factor, depth
    if b == 1:
        return d + 1
    return (b ** (d + 1) - 1) // (b - 1)


def build_tree(branching_factor: int, depth: int) -> list[int]:
    """
    Build a complete b-ary tree with BFS-ordered node indices.

    Returns
    -------
    parent_array : list[int]
        parent_array[i] = index of i's parent; parent_array[0] = -1 (root).
    """
    b, d = branching_factor, depth
    if b <= 0:
        raise ValueError(f"branching_factor must be >= 1, got {b}")
    if d < 0:
        raise ValueError(f"depth must be >= 0, got {d}")

    parent_array: list[int] = [-1]  # root has no parent
    frontier: list[int] = [0]

    for _ in range(d):
        next_frontier: list[int] = []
        for node in frontier:
            for _ in range(b):
                child = len(parent_array)
                parent_array.append(node)
                next_frontier.append(child)
        frontier = next_frontier

    return parent_array


def ancestors(node: int, parent_array: list[int]) -> set[int]:
    """
    Return the set of all ancestors of *node* (inclusive of *node* itself).
    Ancestors are the nodes on the path from root to *node*.
    """
    result: set[int] = set()
    cur = node
    while cur != -1:
        result.add(cur)
        cur = parent_array[cur]
    return result


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

def tree_attention_mask(branching_factor: int, depth: int) -> np.ndarray:
    """
    Construct the causal attention mask for the draft-tree portion only.

    Returns
    -------
    mask : np.ndarray, shape (N, N), dtype bool
        mask[i, j] == True  iff draft token i may attend to draft token j.
        This is True exactly when j ∈ ancestors(i) (including j == i).
    """
    parent_array = build_tree(branching_factor, depth)
    N = len(parent_array)
    mask = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for anc in ancestors(i, parent_array):
            mask[i, anc] = True
    return mask


def tree_attention_mask_n(branching_factor: int, N: int) -> np.ndarray:
    """
    Ancestor attention mask for the **first N BFS nodes** of a b-ary tree.

    Use this when you want a fixed token budget (e.g. Eagle-3's total_token
    of 34–93 tokens), rather than the full complete-tree mask which can be
    astronomically large (O(b^d) × O(b^d) for b=8-12, d=5-9).

    In a BFS-ordered b-ary tree the parent of node i is ``(i - 1) // b`` for
    i > 0 and -1 for the root.  This is verifiably equivalent to the formula
    used by ``build_tree`` / ``tree_attention_mask``.

    Parameters
    ----------
    branching_factor : int
        b — number of children per node.
    N : int
        Number of BFS-ordered tree nodes to include.

    Returns
    -------
    mask : np.ndarray, shape (N, N), dtype bool
        mask[i, j] == True  iff j is an ancestor of i (or j == i),
        restricted to indices 0 … N-1.
    """
    b = branching_factor
    # parent(i) = (i-1)//b  for i > 0;  -1 for i == 0
    parent = np.empty(N, dtype=np.int32)
    parent[0] = -1
    if N > 1:
        idx = np.arange(1, N, dtype=np.int32)
        parent[1:] = (idx - 1) // b

    mask = np.zeros((N, N), dtype=bool)
    for i in range(N):
        j = i
        while j >= 0:
            mask[i, j] = True
            j = int(parent[j])
    return mask


def full_sequence_mask(ctx_len: int, branching_factor: int, depth: int) -> np.ndarray:
    """
    Full attention mask for [context | draft-tree] combined sequence.

    Positions:  [0 .. ctx_len-1]  = context (causal among themselves)
                [ctx_len .. ctx_len+N-1] = draft tokens (tree-causal + full context)

    Returns
    -------
    mask : np.ndarray, shape (ctx_len + N, ctx_len + N), dtype bool
        True  -> token may attend
        False -> masked out
    """
    N = num_tree_nodes(branching_factor, depth)
    total = ctx_len + N
    mask = np.zeros((total, total), dtype=bool)

    # Context tokens: standard lower-triangular causal
    for i in range(ctx_len):
        mask[i, : i + 1] = True

    # Draft tokens: attend to all context + tree ancestors
    tree_mask = tree_attention_mask(branching_factor, depth)
    parent_array = build_tree(branching_factor, depth)
    for i in range(N):
        # Full attend to context
        mask[ctx_len + i, :ctx_len] = True
        # Tree-causal within draft
        for j in range(N):
            mask[ctx_len + i, ctx_len + j] = tree_mask[i, j]

    return mask


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def _tree_depth_of_node(node: int, parent_array: list[int]) -> int:
    depth = 0
    cur = node
    while parent_array[cur] != -1:
        depth += 1
        cur = parent_array[cur]
    return depth


def verify_tree_mask(branching_factor: int, depth: int, verbose: bool = True) -> bool:
    """
    Verify structural properties of the generated tree attention mask.

    Checks
    ------
    1. Reflexivity  : mask[i, i] == True  for all i
    2. Ancestor rule: mask[i, j] == True  iff j is an ancestor of i (incl. j==i)
    3. Anti-sibling : siblings never attend to each other
    4. Count check  : each node i at tree-depth k can see exactly (k+1) tokens
                      (itself + k ancestors)

    Returns True if all checks pass.
    """
    parent_array = build_tree(branching_factor, depth)
    mask = tree_attention_mask(branching_factor, depth)
    N = len(parent_array)
    ok = True

    for i in range(N):
        anc_set = ancestors(i, parent_array)
        node_depth = _tree_depth_of_node(i, parent_array)

        # Check 1: reflexivity
        if not mask[i, i]:
            if verbose:
                print(f"  FAIL reflexivity: mask[{i},{i}] is False")
            ok = False

        # Check 2 & 3: correct ancestor set
        for j in range(N):
            expected = j in anc_set
            actual = bool(mask[i, j])
            if actual != expected:
                if verbose:
                    print(
                        f"  FAIL mask[{i},{j}]: got {actual}, expected {expected} "
                        f"(ancestors of {i} = {anc_set})"
                    )
                ok = False

        # Check 4: row sum equals tree depth + 1
        row_sum = int(mask[i].sum())
        expected_sum = node_depth + 1
        if row_sum != expected_sum:
            if verbose:
                print(
                    f"  FAIL row-sum node {i} depth {node_depth}: "
                    f"got {row_sum}, expected {expected_sum}"
                )
            ok = False

    if verbose:
        status = "PASS" if ok else "FAIL"
        print(
            f"[verify_tree_mask] b={branching_factor} d={depth} "
            f"N={N}  →  {status}"
        )
    return ok


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Tree mask verification ===")
    for b, d in [(1, 4), (2, 3), (3, 2), (4, 2)]:
        verify_tree_mask(b, d, verbose=True)

    print("\n=== Sample mask b=2, d=2 ===")
    mask = tree_attention_mask(2, 2)
    parent = build_tree(2, 2)
    print(f"parent_array: {parent}")
    print(f"mask ({mask.shape[0]}×{mask.shape[1]}):")
    for row in mask.astype(int):
        print(" ", row.tolist())
