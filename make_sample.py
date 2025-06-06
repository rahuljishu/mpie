# make_sample.py
"""
Usage:
    python make_sample.py --data path/to/dataset.csv --rows 10000

Creates a <basename>_10k.csv (or _<rows>k.csv) file inside replay_samples/.
Delimiter is auto-sniffed using utils.loader.load_table.
"""

import argparse, os
from utils.loader  import load_table
from utils.cleaner import clean_dataframe

def main(path: str, rows: int):
    df = clean_dataframe(load_table(path))
    if len(df) > rows:
        df = df.sample(rows, random_state=42)
    basename = os.path.splitext(os.path.basename(path))[0]
    out_dir  = "replay_samples"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{basename}_{rows//1000}k.csv"
    df.to_csv(out_path, index=False)
    print(f"✓ Saved sample to {out_path}  (rows={len(df)})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="CSV/TXT/Excel file")
    p.add_argument("--rows", type=int, default=10_000,
                   help="Number of rows in the sample")
    args = p.parse_args()
    main(args.data, args.rows)
