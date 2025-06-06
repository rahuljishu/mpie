# train.py  – additive / continual version
import argparse, os
from stable_baselines3 import PPO

from envs.mpiejitj_env import MPIEIITJEnv
from utils.loader  import load_table
from utils.cleaner import clean_dataframe


def train(data_path: str,
          timesteps: int = 100_000,
          maxrows: int | None = None,
          progress_bar: bool = False):

    # 1️⃣ load + clean --------------------------------------------------
    df = clean_dataframe(load_table(data_path))
    if maxrows and len(df) > maxrows:
        df = df.sample(maxrows, random_state=42)

    # 2️⃣ create env ----------------------------------------------------
    env = MPIEIITJEnv(df, poly_degree=3)

    # 3️⃣ load-if-exists, else create ----------------------------------
    if os.path.exists("MPIEIITJ_agent.zip"):
        model = PPO.load("MPIEIITJ_agent.zip", env=env)
        print("✓ Loaded existing checkpoint — continuing learning")
    else:
        model = PPO("MlpPolicy", env, verbose=1)
        print("✓ No checkpoint found — training from scratch")

    # 4️⃣ learn ----------------------------------------------------------
    model.learn(total_timesteps=timesteps, progress_bar=progress_bar)

    # 5️⃣ save (overwrites / creates) -----------------------------------
    model.save("MPIEIITJ_agent.zip")
    env.close()


# --------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        required=True,
                        help="Path to CSV/TXT/Excel file")
    parser.add_argument("--timesteps",   type=int, default=100_000)
    parser.add_argument("--maxrows",     type=int,
                        help="Randomly sample N rows (optional)")
    parser.add_argument("--progress_bar", action="store_true",
                        help="Show tqdm progress bar")
    args = parser.parse_args()

    train(args.data, args.timesteps, args.maxrows, args.progress_bar)
