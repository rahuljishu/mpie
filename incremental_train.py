# incremental_train.py  – fine-tune on NEW data without forgetting
import os, glob, random, argparse
from stable_baselines3 import PPO
from utils.loader  import load_table
from utils.cleaner import clean_dataframe
from envs.mpiejitj_env import MPIEIITJEnv

def incremental_update(new_file: str,
                       epochs: int = 10,
                       steps_per_epoch: int = 10_000,
                       replay_dir: str = "replay_samples",
                       progress_bar: bool = True):

    # 1️⃣ make sure master checkpoint exists
    if not os.path.exists("MPIEIITJ_agent.zip"):
        raise FileNotFoundError("Run train.py at least once before fine-tuning.")

    # 2️⃣ load NEW data frame
    new_df = clean_dataframe(load_table(new_file))

    # 3️⃣ build replay pool  (create dir if first time)
    os.makedirs(replay_dir, exist_ok=True)
    replay_paths = glob.glob(f"{replay_dir}/*.csv")
    replay_dfs   = [clean_dataframe(load_table(p)) for p in replay_paths]

    # 4️⃣ load model with ANY env (dummy) then fine-tune in loops
    dummy_env = MPIEIITJEnv(new_df)        # give action/obs spaces
    model = PPO.load("MPIEIITJ_agent.zip", env=dummy_env)
    print(f"✓ Loaded master checkpoint  (replay pool = {len(replay_dfs)} files)")

    for _ in range(epochs):
        df_batch = random.choice(replay_dfs + [new_df]) if replay_dfs else new_df
        env_batch = MPIEIITJEnv(df_batch)
        model.set_env(env_batch)
        model.learn(total_timesteps=steps_per_epoch, progress_bar=progress_bar)

    # 5️⃣ save updated master
    model.save("MPIEIITJ_agent.zip")

    # 6️⃣ add a capped sample of the new dataset to replay pool
    sample_rows = min(10_000, len(new_df))
    sample_path = f"{replay_dir}/{os.path.basename(new_file)}_{sample_rows//1000}k.csv"
    new_df.sample(sample_rows, random_state=42).to_csv(sample_path, index=False)
    print(f"✓ Added replay sample → {sample_path}  (rows={sample_rows})")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to NEW dataset")
    p.add_argument("--epochs", type=int, default=10, help="Fine-tune epochs")
    p.add_argument("--steps",  type=int, default=10_000, help="Steps/epoch")
    args = p.parse_args()
    incremental_update(args.data, epochs=args.epochs, steps_per_epoch=args.steps)
