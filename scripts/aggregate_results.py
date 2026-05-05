#!/usr/bin/env python3
import os
import glob
import pandas as pd
import re

def get_stable_name(fname):
    stem, ext = os.path.splitext(fname)
    stem = re.sub(r'_\d{8}_\d{6}$', '', stem)
    return stem + ext

def aggregate_experiment(base_dir):
    runs = sorted(glob.glob(os.path.join(base_dir, "run_*")))
    if not runs:
        return
        
    stable_names = set()
    for run in runs:
        for f in glob.glob(os.path.join(run, "*.csv")):
            stable_names.add(get_stable_name(os.path.basename(f)))
            
    for sname in stable_names:
        dfs = []
        for run in runs:
            stem = sname.replace('.csv', '')
            matches = glob.glob(os.path.join(run, f"{stem}*.csv"))
            if matches:
                # Prefer pruned over plain
                best_match = next((m for m in matches if 'pruned' in m), matches[0])
                dfs.append(pd.read_csv(best_match))
        
        if not dfs:
            continue
            
        combined = pd.concat(dfs)
        
        configs = []
        metrics = []
        
        # Identify columns
        # Configuration parameters are typically integers, strings, or booleans.
        # Runtimes and metrics are floats.
        for col in dfs[0].columns:
            if pd.api.types.is_float_dtype(dfs[0][col]):
                metrics.append(col)
            else:
                configs.append(col)
                
        if not metrics:
            print(f"  [aggregate] Warning: No float metrics found in {sname}. Skipping.")
            continue
            
        # Group by configuration parameters and average the metrics.
        # This keeps the exact configuration parameters (without averaging them)
        # and leaves the metric columns with their original names.
        agg_df = combined.groupby(configs, dropna=False)[metrics].mean().reset_index()
        
        # Reorder columns to match the original structure
        agg_df = agg_df[dfs[0].columns]
            
        out_name = sname.replace('.csv', '_aggregate.csv')
        out_path = os.path.join(base_dir, out_name)
        agg_df.to_csv(out_path, index=False)
        print(f"  [aggregate] {sname} -> {out_path} (Configs: {len(configs)}, Metrics: {len(metrics)})")

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(repo_root, "results")
    
    for exp in ["micro", "sequoia", "eagle_e2e", "amdahl"]:
        exp_dir = os.path.join(results_dir, exp)
        if os.path.isdir(exp_dir):
            aggregate_experiment(exp_dir)
            
if __name__ == "__main__":
    main()
