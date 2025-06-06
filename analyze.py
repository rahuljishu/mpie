# analyze.py
import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from stable_baselines3 import PPO

from envs.mpiejitj_env import MPIEIITJEnv
from utils.loader  import load_table
from utils.cleaner import clean_dataframe

MODELPATH = "MPIEIITJ_agent.zip"


# ----------------------------------------------------------------------
def find_best_relations(df: pd.DataFrame,
                        max_degree: int = 3,
                        top_k: int = 5):
    """Return the strongest polynomial relations (R²) among all numeric pairs."""
    numeric = df.select_dtypes(include=[np.number]).dropna()
    rels = []
    for c1, c2 in combinations(numeric.columns, 2):
        X, y = numeric[c1].values.reshape(-1, 1), numeric[c2].values
        best = {"pair": (c1, c2), "degree": None, "r2": -np.inf}
        for deg in range(1, max_degree + 1):
            Xp = PolynomialFeatures(deg).fit_transform(X)
            r2 = LinearRegression().fit(Xp, y).score(Xp, y)
            if r2 > best["r2"]:
                best.update({"degree": deg, "r2": r2})
        rels.append(best)
    return sorted(rels, key=lambda d: d["r2"], reverse=True)[:top_k]


# ----------------------------------------------------------------------
def analyze(data_path: str):
    # 1️⃣ load + clean
    df = clean_dataframe(load_table(data_path))

    # 2️⃣ load trained policy once
    model = PPO.load(MODELPATH)

    # 3️⃣ evaluate every numeric column separately ----------------------
    env = MPIEIITJEnv(df)
    best = {"col": None, "tr": None, "rew": -np.inf, "info": None}

    for col in env.cols:
        env.current_col = col
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, rew, _, _, info = env.step(action)

        if rew > best["rew"]:
            best.update(col=col, tr=info["transform"], rew=rew, info=info)

    env.close()

    print(f"\nBest column: {best['col']} "
          f"(transform={best['tr']}, reward={best['rew']:.3f})")
    print("Reward break-down:",
          {k: round(v, 3) for k, v in best["info"].items()
           if k not in ("column", "transform")})

    # 4️⃣ pair-wise polynomial relations --------------------------------
    print("\nTop relations:")
    for rel in find_best_relations(df):
        a, b = rel["pair"]
        print(f"{a} → {b}  deg={rel['degree']}  R²={rel['r2']:.3f}")

    # 5️⃣ global PCA summary — drop zero-variance columns first -------
    num_df = df.select_dtypes(include=[np.number]).dropna()

    # compute per-column std and keep only those > 0
    vars_    = num_df.std(axis=0)
    nonconst = vars_[vars_ > 0].index.tolist()

    if len(nonconst) >= 2:
        sub = num_df[nonconst]
        pca = PCA(n_components=2).fit(sub.values)
        print("\nPCA variance ratios:", pca.explained_variance_ratio_.round(3))
    else:
        print("\nPCA variance ratios: not enough non-constant columns")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True,
                        help="Path to CSV / TXT / Excel file")
    args = parser.parse_args()
    analyze(args.data)
