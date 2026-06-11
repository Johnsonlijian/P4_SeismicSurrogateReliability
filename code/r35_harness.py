"""Shared harness for the R35 experiments (E1 partition stability, E2 Sa(T1)
ablation, E3 bound coverage).

Faithfully reproduces the submitted pipeline:
- features/splits as in 23_nonlinear_mdof_label_budget.py (seeds 20260526/27;
  protocol rows are the T1 >= 1.8 s target systems; source pretraining rows
  are T1 <= 1.0 s),
- the original event-disjoint partition as in 66_* (seed 20260613),
- model hyperparameters via direct import of 02_train_models.py,
- conformal split (half fit / half calibration) and the (n+1) order-statistic
  quantile as in 52_*/67_*.

Adds: optional spectral features Sa(T1) and Sa_avg(0.2T1-3T1) computed from
the stored ground-motion array (linear SDOF, Newmark average acceleration,
zeta = 5%), and event-equal decision metrics with event-bootstrap bounds and
conditional miss rates.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REC_DIR = ROOT / "outputs" / "high_target" / "recorded_nsmp_full"
MDOF_DIR = ROOT / "outputs" / "high_target" / "nonlinear_mdof_grid_full"

GM_FEATURES = ["pga_g", "pgv_m_s", "pgd_m", "arias_m_s", "cav_m_s",
               "d5_95_s", "tm_s", "pgv_pga_s"]
SYS_FEATURES = ["n_story", "T1_s", "zeta", "yield_drift", "alpha_post_yield",
                "is_soft_first_story"]
SPECTRAL_FEATURES = ["sa_t1_g", "sa_avg_g"]

LOG1PCT = math.log10(0.01)
NORMAL = NormalDist()
G = 9.81
DT = 0.01

_spec = importlib.util.spec_from_file_location("train_mod", HERE / "02_train_models.py")
train_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(train_mod)


# ----------------------------------------------------------------- data -----
def load_records() -> pd.DataFrame:
    return pd.read_csv(REC_DIR / "nsmp_recorded_records.csv")


def load_labels() -> pd.DataFrame:
    df = pd.read_csv(MDOF_DIR / "recorded_nonlinear_mdof_labels.csv")
    df = df[df["is_valid"] == True].copy()  # noqa: E712
    df["is_soft_first_story"] = (df["pattern"] == "soft_first_story").astype(float)
    return df


def split_events(records: pd.DataFrame, target_fraction: float = 0.35):
    """Source/target event split + main-protocol component split (orig seeds)."""
    events = np.array(sorted(records["event_id"].unique()))
    rng = np.random.default_rng(20260526)
    perm = rng.permutation(events)
    n_target = max(3, int(round(len(events) * target_fraction)))
    target_events = sorted(perm[:n_target].tolist())
    source_events = sorted(perm[n_target:].tolist())
    target_gms = records[records["event_id"].isin(target_events)]["gm_id"].to_numpy(int)
    rng2 = np.random.default_rng(20260527)
    target_perm = rng2.permutation(target_gms)
    split = len(target_perm) // 2
    pool_gm = sorted(target_perm[:split].tolist())
    test_gm = sorted(target_perm[split:].tolist())
    source_gm = sorted(records[records["event_id"].isin(source_events)]["gm_id"].to_numpy(int).tolist())
    return source_events, target_events, source_gm, pool_gm, test_gm


def disjoint_partition(records: pd.DataFrame, target_events: list[str],
                       seed: int = 20260613):
    """7 fit / 8 test event partition of the 15 target events (orig rule)."""
    arr = np.array(sorted(target_events))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(arr)
    cut = max(1, len(perm) // 2)
    fit_events = sorted(perm[:cut].tolist())
    test_events = sorted(perm[cut:].tolist())
    fit_gm = sorted(records[records["event_id"].isin(fit_events)]["gm_id"].to_numpy(int).tolist())
    test_gm = sorted(records[records["event_id"].isin(test_events)]["gm_id"].to_numpy(int).tolist())
    return fit_gm, test_gm, fit_events, test_events


def merge_features(labels: pd.DataFrame, records: pd.DataFrame,
                   gm_ids=None, source_period: str = "all",
                   spectra: pd.DataFrame | None = None):
    df = labels.copy()
    if gm_ids is not None:
        df = df[df["gm_id"].isin(gm_ids)]
    if source_period == "seen":
        df = df[df["T1_s"] <= 1.0]
    elif source_period == "target":
        df = df[df["T1_s"] >= 1.8]
    record_cols = [c for c in GM_FEATURES if c not in df.columns]
    df = df.merge(records[["gm_id", "event_id"] + record_cols], on="gm_id")
    features = GM_FEATURES + SYS_FEATURES
    if spectra is not None:
        df = df.merge(spectra, on=["gm_id", "T1_s"], how="left")
        features = features + SPECTRAL_FEATURES
    df = df.reset_index(drop=True)
    return df, df[features].to_numpy(float), df["log_max_idr"].to_numpy(float)


# -------------------------------------------------------------- spectra -----
def _newmark_sd(acc: np.ndarray, dt: float, periods: np.ndarray,
                zeta: float = 0.05) -> np.ndarray:
    """Peak relative displacement of linear SDOFs (Newmark avg acceleration).

    acc: (n_rec, n_steps) ground acceleration in m/s^2.
    Returns (n_rec, n_periods) peak |u| in m. Vectorized over records.
    """
    n_rec, n_steps = acc.shape
    out = np.zeros((n_rec, len(periods)))
    gamma, beta = 0.5, 0.25
    for j, T in enumerate(periods):
        wn = 2.0 * np.pi / T
        k = wn ** 2
        c = 2.0 * zeta * wn
        a1 = 1.0 / (beta * dt ** 2) + gamma * c / (beta * dt)
        a2 = 1.0 / (beta * dt) + (gamma / beta - 1.0) * c
        a3 = (1.0 / (2.0 * beta) - 1.0) + dt * c * (gamma / (2.0 * beta) - 1.0)
        kh = k + a1
        u = np.zeros(n_rec)
        v = np.zeros(n_rec)
        a = -acc[:, 0]
        peak = np.zeros(n_rec)
        p = -acc
        for i in range(1, n_steps):
            ph = p[:, i] + a1 * u + a2 * v + a3 * a
            u_new = ph / kh
            v_new = (gamma / (beta * dt)) * (u_new - u) + (1.0 - gamma / beta) * v \
                + dt * (1.0 - gamma / (2.0 * beta)) * a
            a_new = (u_new - u) / (beta * dt ** 2) - v / (beta * dt) \
                - (1.0 / (2.0 * beta) - 1.0) * a
            u, v, a = u_new, v_new, a_new
            np.maximum(peak, np.abs(u), out=peak)
        out[:, j] = peak
    return out


def compute_spectra(cache_path: Path | None = None) -> pd.DataFrame:
    """Sa(T1) and Sa_avg(0.2T1-3T1) in g for T1 in {0.5, 1.0, 1.8} s."""
    if cache_path is not None and cache_path.exists():
        return pd.read_csv(cache_path)
    acc = np.load(REC_DIR / "nsmp_recorded_gm_array.npy")  # m/s^2, gm_id order
    t1_values = [0.5, 1.0, 1.8]
    rows_per_t1 = {}
    for T1 in t1_values:
        grid = np.geomspace(0.2 * T1, 3.0 * T1, 16)
        periods = np.unique(np.concatenate([[T1], grid]))
        sd = _newmark_sd(acc, DT, periods)
        psa = sd * (2.0 * np.pi / periods[None, :]) ** 2 / G  # pseudo-Sa in g
        i_t1 = int(np.where(np.isclose(periods, T1))[0][0])
        sa_t1 = psa[:, i_t1]
        in_grid = np.isin(periods, grid)
        sa_avg = np.exp(np.mean(np.log(np.clip(psa[:, in_grid], 1e-8, None)), axis=1))
        rows_per_t1[T1] = (sa_t1, sa_avg)
    recs = []
    n_rec = acc.shape[0]
    for T1 in t1_values:
        sa_t1, sa_avg = rows_per_t1[T1]
        for gm in range(n_rec):
            recs.append({"gm_id": gm, "T1_s": T1,
                         "sa_t1_g": float(sa_t1[gm]), "sa_avg_g": float(sa_avg[gm])})
    df = pd.DataFrame(recs)
    if cache_path is not None:
        df.to_csv(cache_path, index=False)
    return df


# --------------------------------------------------------------- models -----
def fit_and_predict(model_name: str, X_train, y_train, X_calib, X_test,
                    seed: int, foundation=None, sc=None):
    if model_name == "ridge_direct":
        m, s = train_mod.fit_ridge_fewshot(X_train, y_train)
        return m.predict(s.transform(X_calib)), m.predict(s.transform(X_test))
    if model_name == "rf_direct":
        m = train_mod.fit_rf_fewshot(X_train, y_train, seed)
        return m.predict(X_calib), m.predict(X_test)
    if model_name == "hgb_direct":
        m = train_mod.fit_histgb_fewshot(X_train, y_train, seed)
        return m.predict(X_calib), m.predict(X_test)
    if model_name == "xgb_direct":
        m = train_mod.fit_xgb_fewshot(X_train, y_train, seed)
        return m.predict(X_calib), m.predict(X_test)
    if model_name == "lgbm_direct":
        m = train_mod.fit_lgbm_fewshot(X_train, y_train, seed)
        return m.predict(X_calib), m.predict(X_test)
    if model_name == "scratch_mlp":
        m, s = train_mod.fit_baseline(X_train, y_train, seed)
        return m.predict(s.transform(X_calib)), m.predict(s.transform(X_test))
    if model_name == "pretrained_finetune":
        assert foundation is not None and sc is not None
        m = train_mod.finetune(foundation, sc, X_train, y_train, seed)
        return m.predict(sc.transform(X_calib)), m.predict(sc.transform(X_test))
    raise ValueError(model_name)


def conformal_quantile(abs_resid: np.ndarray, alpha: float = 0.10) -> float:
    vals = np.sort(np.asarray(abs_resid, float))
    n = len(vals)
    k = min(max(int(math.ceil((1.0 - alpha) * (n + 1))), 1), n)
    return float(vals[k - 1])


# -------------------------------------------------------------- metrics -----
def beta_from_p(p: float) -> float:
    p = min(max(float(p), 1e-9), 1 - 1e-9)
    return float(-NORMAL.inv_cdf(p))


def event_equal_mean(flags: np.ndarray, events: np.ndarray) -> float:
    return float(pd.DataFrame({"f": flags.astype(float), "e": events})
                 .groupby("e")["f"].mean().mean())


def event_boot_upper(flags: np.ndarray, events: np.ndarray,
                     rng: np.random.Generator, n_boot: int = 2000) -> float:
    means = (pd.DataFrame({"f": flags.astype(float), "e": events})
             .groupby("e")["f"].mean().to_numpy())
    draws = rng.choice(means, size=(n_boot, len(means)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.95))


def decision_metrics(y_true: np.ndarray, y_pred: np.ndarray, q: float,
                     events: np.ndarray, rng: np.random.Generator,
                     log_thr: float = LOG1PCT) -> dict[str, float]:
    upper = y_pred + q
    unsafe = y_true > log_thr
    pred_safe = upper <= log_thr
    fs = unsafe & pred_safe
    fu = (~unsafe) & (~pred_safe)
    covered = np.abs(y_true - y_pred) <= q
    pfs = event_equal_mean(fs, events)
    pfs_hi = event_boot_upper(fs, events, rng)
    if unsafe.sum():
        ue = pd.DataFrame({"m": fs[unsafe].astype(float), "e": events[unsafe]})
        cond_event_mean = float(ue.groupby("e")["m"].mean().mean())
        cond_pooled = float(fs[unsafe].mean())
    else:
        cond_event_mean = cond_pooled = float("nan")
    return {
        "q_log10": float(q),
        "rmse_log": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "coverage_event_equal": event_equal_mean(covered, events),
        "p_false_safe": pfs,
        "p_false_safe_u95": pfs_hi,
        "beta_fs_cons": beta_from_p(pfs_hi),
        "p_false_unsafe": event_equal_mean(fu, events),
        "conditional_miss_event_mean": cond_event_mean,
        "conditional_miss_pooled": cond_pooled,
        "n_unsafe_rows": int(unsafe.sum()),
        "n_events_with_unsafe": int(pd.unique(events[unsafe]).size),
        "n_test_rows": int(len(y_true)),
        "n_test_events": int(pd.unique(events).size),
    }


def run_cell(model_name: str, X_pool, y_pool, X_test, y_test, test_events,
             N: int, rep_seed: int, rng_boot: np.random.Generator,
             alpha: float = 0.10, foundation=None, sc=None) -> dict[str, float]:
    """One (model, budget, rep): sample N, half fit / half calibrate, score."""
    rng = np.random.default_rng(rep_seed)
    sel = rng.choice(np.arange(len(X_pool)), size=N, replace=False)
    X_sel, y_sel = X_pool[sel], y_pool[sel]
    half = N // 2
    perm = rng.permutation(N)
    tr, ca = perm[:half], perm[half:]
    pred_calib, pred_test = fit_and_predict(
        model_name, X_sel[tr], y_sel[tr], X_sel[ca], X_test, rep_seed,
        foundation=foundation, sc=sc)
    q = conformal_quantile(np.abs(y_sel[ca] - pred_calib), alpha)
    return decision_metrics(y_test, pred_test, q, test_events, rng_boot)
