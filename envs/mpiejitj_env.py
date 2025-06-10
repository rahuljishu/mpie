# envs/mpiejitj_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from scipy.fftpack import dct, fft
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pywt                          # PyWavelets==1.4.1 wheel


class MPIEIITJEnv(gym.Env):
    """
    Column-agnostic, one-step episode:
      • env chooses ONE numeric column per episode
      • observation = 4 stats of that column
      • action      = which transform (PCA/DCT/Wavelet/FFT/Polyfit)
      • reward      = weighted sparsity + symmetry + eq_fit + compression
    Works for ANY dataset size because obs/action size is fixed.
    """

    metadata = {"render_modes": []}

    # ------------------------------------------------------------------ #
    def __init__(self, df: pd.DataFrame, poly_degree: int = 3):
        super().__init__()

        # 1️⃣  numeric only
        df_num = df.copy()
        for col in df_num.select_dtypes(include=["object", "category"]):
            df_num[col], _ = pd.factorize(df_num[col])

        df_num = (df_num.select_dtypes(include=[np.number])
                           .replace([np.inf, -np.inf], np.nan)
                           .dropna(axis=1, how="all"))

        self.df   = df_num
        self.cols = list(df_num.columns)
        if not self.cols:
            raise ValueError("No numeric columns found after cleaning the dataset")

        # 2️⃣  fixed transform list (5 actions)
        self.transforms = ["pca", "dct", "wavelet", "fft", "polyfit"]

        # 3️⃣  Gymnasium spaces  ⬅︎  changed
        self.action_space      = spaces.Discrete(len(self.transforms))  # =5
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(4,), dtype=np.float32)

        self.poly_degree = poly_degree
        self.state       = None
        self.current_col = None

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # pick ONE random column each episode  ⬅︎  changed
        self.current_col = self.np_random.choice(self.cols)
        data = self.df[self.current_col].values.astype(np.float32)

        if data.size == 0 or np.all(np.isnan(data)):
            mean = std = skew = kurt = 0.0
        else:
            mean = np.nanmean(data)
            std  = max(np.nanstd(data), 1e-12)
            skew = np.nanmean(((data - mean) / std) ** 3)
            kurt = np.nanmean(((data - mean) / std) ** 4) - 3

        self.state = np.nan_to_num(np.array([mean, std, skew, kurt],
                                            dtype=np.float32))
        return self.state, {}

    # ------------------------------------------------------------------ #
    def step(self, action: int):
        name = self.transforms[action]          # ⬅︎  action → transform
        col  = self.current_col
        data = self.df[col].values.astype(np.float32)

        coeffs   = self._apply_transform(data, name)
        thr      = np.mean(np.abs(coeffs))
        sparsity = 1.0 - (np.sum(np.abs(coeffs) > thr) / coeffs.size)

        sym = np.corrcoef(data, np.flip(data))[0, 1]
        sym = 0.0 if np.isnan(sym) else sym

        eq_score = 0.0
        if name == "polyfit":
            X    = np.arange(len(data)).reshape(-1, 1)
            poly = PolynomialFeatures(self.poly_degree)
            reg  = LinearRegression().fit(poly.fit_transform(X), data)
            eq_score = reg.score(poly.transform(X), data)

        comp_ratio = PCA(n_components=1).fit(data.reshape(-1, 1)) \
                                         .explained_variance_ratio_[0]

        sparsity, sym, eq_score, comp_ratio = np.nan_to_num(
            [sparsity, sym, eq_score, comp_ratio])

        reward = (0.4 * sparsity + 0.2 * sym +
                  0.3 * eq_score + 0.1 * comp_ratio)

        info = dict(column=col, transform=name,
                    sparsity=sparsity, symmetry=sym,
                    equation_score=eq_score,
                    compression_ratio=comp_ratio)

        return self.state, float(reward), True, False, info

    # ------------------------------------------------------------------ #
    def _apply_transform(self, data: np.ndarray, name: str) -> np.ndarray:
        if name == "pca":
            return PCA(n_components=1).fit_transform(data[:, None]).ravel()
        if name == "dct":
            return dct(data, norm="ortho")
        if name == "wavelet":
            coeffs, _ = pywt.dwt(data, "db1")
            return coeffs
        if name == "fft":
            return np.abs(fft(data))
        if name == "polyfit":
            X    = np.arange(len(data)).reshape(-1, 1)
            poly = PolynomialFeatures(self.poly_degree)
            pred = LinearRegression().fit(poly.fit_transform(X), data) \
                                     .predict(poly.transform(X))
            return data - pred
        raise ValueError(f"Unknown transform {name}")
