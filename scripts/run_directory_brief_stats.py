#!/usr/bin/env python3
"""Write a brief statistical summary CSV per results suite.

The script scans results/*/run_* directories, combines the matching CSVs across
all runs in each suite, and writes a one-row brief summary into the suite parent
directory. The summary is analogous to the aggregate script, but instead of
emitting the full merged tables it reports across-run statistics such as mean,
median, and standard deviation for each numeric metric.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Iterable, Sequence

import pandas as pd


DEFAULT_METRICS = [
    "wall_ms",
    "tok_per_sec",
    "mean_accepted_per_step",
    "acceptance_rate",
    "mean_verify_ms",
    "verify_fraction",
    "speedup_vs_tree",
    "speedup_vs_cascade",
    "speedup_vs_deft",
    "e2e_speedup",
    "ragged_ms",
    "flashinfer_tree_ms",
    "flashinfer_cascade_ms",
    "deft_ms",
]


def _stable_name(path: str) -> str:
    """Strip trailing _YYYYMMDD_HHMMSS from a CSV basename."""
    fname = os.path.basename(path)
    stem, ext = os.path.splitext(fname)
    stem = re.sub(r"_\d{8}_\d{6}$", "", stem)
    return stem + ext


def _is_skip_csv(path: str) -> bool:
    base = os.path.basename(path)
    return base == "aggregate.csv" or base.endswith("_aggregate.csv")


def _aggregate_suite_csvs(csv_paths: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Group run CSVs by stable basename and merge each group."""
    grouped: dict[str, list[pd.DataFrame]] = {}
    for path in csv_paths:
        if _is_skip_csv(path):
            continue
        grouped.setdefault(_stable_name(path), []).append(pd.read_csv(path))

    merged: dict[str, pd.DataFrame] = {}
    for name, frames in grouped.items():
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        config_cols = [c for c in df.columns if not pd.api.types.is_float_dtype(df[c])]
        metric_cols = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]

        if metric_cols and config_cols:
            merged[name] = df.groupby(config_cols, dropna=False)[metric_cols].agg(["mean", "median", "std"])
        elif metric_cols:
            merged[name] = df[metric_cols].agg(["mean", "median", "std"])
        else:
            merged[name] = pd.DataFrame()

    return merged


def _summarize_merged_frames(merged: dict[str, pd.DataFrame], metrics: Sequence[str]) -> dict:
    summary: dict[str, object] = {}
    for section_name, df in merged.items():
        summary[f"{section_name}__n_groups"] = int(len(df))

        if isinstance(df.columns, pd.MultiIndex):
            for metric in metrics:
                if metric not in df.columns.get_level_values(0):
                    continue

                series_mean = pd.to_numeric(df[(metric, "mean")], errors="coerce").dropna()
                series_median = pd.to_numeric(df[(metric, "median")], errors="coerce").dropna()
                series_std = pd.to_numeric(df[(metric, "std")], errors="coerce").dropna()

                if not series_mean.empty:
                    summary[f"{section_name}__{metric}__mean"] = float(series_mean.mean())
                    summary[f"{section_name}__{metric}__median"] = float(series_mean.median())
                    summary[f"{section_name}__{metric}__std"] = float(series_mean.std())
                if not series_median.empty:
                    summary[f"{section_name}__{metric}__group_median_mean"] = float(series_median.mean())
                if not series_std.empty:
                    summary[f"{section_name}__{metric}__run_std_mean"] = float(series_std.mean())
        else:
            for metric in metrics:
                if metric not in df.columns:
                    continue
                s = pd.to_numeric(df[metric], errors="coerce").dropna()
                if s.empty:
                    continue
                summary[f"{section_name}__{metric}__mean"] = float(s.mean())
                summary[f"{section_name}__{metric}__median"] = float(s.median())
                summary[f"{section_name}__{metric}__std"] = float(s.std())

    return summary


def _suite_run_csvs(suite_dir: str) -> list[str]:
    csv_paths: list[str] = []
    run_dirs = [d for d in sorted(glob.glob(os.path.join(suite_dir, "run_*"))) if os.path.isdir(d)]

    if run_dirs:
        for run_dir in run_dirs:
            csv_paths.extend([p for p in glob.glob(os.path.join(run_dir, "*.csv")) if not _is_skip_csv(p)])
    else:
        # Flat suites such as A100-ctx-sweep and A100-n-sweep store CSVs
        # directly under the suite directory instead of run_*/ subfolders.
        csv_paths.extend([p for p in glob.glob(os.path.join(suite_dir, "*.csv")) if not _is_skip_csv(p)])

    return csv_paths


def process_suite_dir(suite_dir: str, out_name: str, metrics: Sequence[str], overwrite: bool) -> str | None:
    csv_paths = _suite_run_csvs(suite_dir)
    if not csv_paths:
        return None

    out_path = os.path.join(suite_dir, out_name)
    if os.path.exists(out_path) and not overwrite:
        return out_path

    merged = _aggregate_suite_csvs(csv_paths)
    summary = _summarize_merged_frames(merged, metrics)
    summary["n_run_csvs"] = int(len(csv_paths))
    summary["n_sections"] = int(len(merged))

    out_df = pd.DataFrame([summary])
    out_df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a one-row statistical summary CSV into each results suite directory.",
    )
    parser.add_argument("--results-root", default="results", help="Root results directory")
    parser.add_argument("--out-name", default="brief_stats.csv", help="Output CSV name inside each suite directory")
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metric columns to summarize",
    )
    parser.add_argument("--all-numeric", action="store_true", help="Summarize every numeric-looking column")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    results_root = os.path.abspath(args.results_root)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    n_written = 0
    for suite in sorted(os.listdir(results_root)):
        suite_dir = os.path.join(results_root, suite)
        if not os.path.isdir(suite_dir) or suite == "old":
            continue

        use_metrics = metrics
        if args.all_numeric:
            # Use the union of numeric-looking columns across the suite.
            suite_csvs = _suite_run_csvs(suite_dir)
            numeric_cols: set[str] = set()
            for src in suite_csvs:
                df = pd.read_csv(src)
                for c in df.columns:
                    if pd.to_numeric(df[c], errors="coerce").notna().any():
                        numeric_cols.add(c)
            use_metrics = sorted(numeric_cols)

        out_path = process_suite_dir(suite_dir, args.out_name, use_metrics, args.overwrite)

        if out_path:
            n_written += 1
            print(out_path)

    print(f"Wrote {n_written} brief summary file(s).")


if __name__ == "__main__":
    main()
