import os
import glob
import pandas as pd
import re
import warnings
warnings.filterwarnings("ignore")

def process_dir(d):
    runs = sorted(glob.glob(os.path.join(d, "run_*")))
    if not runs:
        return
    
    all_dfs = {}
    for r in runs:
        csvs = glob.glob(os.path.join(r, "*.csv"))
        # Prioritize files without timestamps if they exist to avoid duplicates
        plain_csvs = [c for c in csvs if not re.search(r'_\d{8}_\d{6}\.csv$', c)]
        target_csvs = plain_csvs if plain_csvs else csvs
        
        for c in target_csvs:
            b = os.path.basename(c)
            name = re.sub(r'_\d{8}_\d{6}', '', b)
            if name not in all_dfs:
                all_dfs[name] = []
            all_dfs[name].append(pd.read_csv(c))
            
    res_dfs = []
    for name, list_df in all_dfs.items():
        if len(list_df) == 0: continue
        df = pd.concat(list_df, ignore_index=True)
        
        config_cols = [c for c in df.columns if not pd.api.types.is_float_dtype(df[c])]
        metrics = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
        
        if len(metrics) > 0 and len(config_cols) > 0:
            df = df.groupby(config_cols, dropna=False)[metrics].mean().reset_index()
        elif len(metrics) > 0:
            df = df.mean().to_frame().T
            
        res = []
        for col in list_df[0].columns:
            if col in df.columns:
                res.append(df[col])
        res_dfs.append(pd.concat(res, axis=1))

    out_path = os.path.join(d, "aggregate.csv")
    if len(res_dfs) >= 1:
        with open(out_path, "w") as f:
            for i, rdf in enumerate(res_dfs):
                rdf.to_csv(f, index=False)
                if i < len(res_dfs) - 1:
                    f.write("\n")

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results = os.path.join(root, "results")
    for d in os.listdir(results):
        d_path = os.path.join(results, d)
        if os.path.isdir(d_path) and d != "old":
            process_dir(d_path)

if __name__ == "__main__":
    main()
