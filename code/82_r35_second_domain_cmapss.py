"""R35-B: second-domain transferability demonstration (NASA C-MAPSS FD002).

Purpose: show the guarded eligibility audit is domain-agnostic by mapping it
onto turbofan remaining-useful-life (RUL) screening: engine units play the
role of earthquake events, the maintenance threshold tau plays the drift
threshold, and the false-safe event is operating an engine whose true RUL is
below tau because the surrogate's conformal LOWER bound overstated it.

Framing for the manuscript: a transferability check of the AUDIT MECHANICS -
explicitly not a contribution to RUL prediction (standard piecewise-capped
RUL target, plain per-cycle features, no PHM state-of-the-art comparison).

Contrast of interest: C-MAPSS train engines are run-to-failure, so every
engine carries unsafe-regime rows - an evidence-RICH grouped population,
the mirror image of the evidence-poor seismic archive. The audit should
therefore reach stable verdicts at small E here, demonstrating that the
seismic-domain instability was evidence-driven, not audit-driven.
"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

import r35_harness as H

OUT = H.ROOT / "outputs" / "high_target" / "r35_cmapss"
ROUND = H.ROOT / "rounds" / "R34_figure_style_normalization_2026-06-10"
URL = ("https://raw.githubusercontent.com/mapr-demos/predictive-maintenance/"
       "master/notebooks/jupyter/Dataset/CMAPSSData/train_FD002.txt")

TAU = 30          # cycles; unsafe = RUL <= TAU
RUL_CAP = 125     # standard piecewise target
N_BUDGET = 2000
N_REPS = 3
N_POOL_ENGINES = 60
E_DRAW = 8
N_DRAWS = 300
N_BOOT = 1000
BETA_TARGET = 2.5
PFU_MAX = 0.30
M_STAR = 0.25
MODELS = ["ridge_direct", "lgbm_direct", "xgb_direct", "scratch_mlp"]
NORMAL = NormalDist()


def load_fd002() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "train_FD002.txt"
    if not cache.exists():
        urllib.request.urlretrieve(URL, cache)
    cols = (["unit", "cycle", "set1", "set2", "set3"]
            + [f"s{i}" for i in range(1, 22)])
    df = pd.read_csv(cache, sep=r"\s+", header=None, names=cols)
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = np.minimum(max_cycle - df["cycle"], RUL_CAP).astype(float)
    return df


def one_sided_quantile(signed_over: np.ndarray, level: float = 0.90) -> float:
    """Order-statistic upper quantile of (pred - true) overprediction."""
    vals = np.sort(np.asarray(signed_over, float))
    n = len(vals)
    k = min(max(int(np.ceil(level * (n + 1))), 1), n)
    return float(vals[k - 1])


def engine_equal(flags: np.ndarray, units: np.ndarray) -> float:
    return float(pd.DataFrame({"f": flags.astype(float), "u": units})
                 .groupby("u")["f"].mean().mean())


def main() -> None:
    t0 = time.time()
    df = load_fd002()
    feats = ["set1", "set2", "set3"] + [f"s{i}" for i in range(1, 22)]
    units = np.array(sorted(df["unit"].unique()))
    rng_split = np.random.default_rng(20260620)
    perm = rng_split.permutation(units)
    pool_units = set(perm[:N_POOL_ENGINES].tolist())
    pop_units_arr = perm[N_POOL_ENGINES:]
    pool = df[df["unit"].isin(pool_units)]
    pop = df[df["unit"].isin(set(pop_units_arr.tolist()))]
    X_pool, y_pool = pool[feats].to_numpy(float), pool["rul"].to_numpy(float)
    X_pop, y_pop = pop[feats].to_numpy(float), pop["rul"].to_numpy(float)
    u_pop = pop["unit"].to_numpy()
    print(f"[82] pool {len(X_pool)} rows / {N_POOL_ENGINES} engines; "
          f"population {len(X_pop)} rows / {len(pop_units_arr)} engines; "
          f"unsafe share {float((y_pop <= TAU).mean()):.3f}", flush=True)

    rep_seeds = [int(s) for s in
                 np.random.default_rng(20260621).integers(1, 10 ** 8, N_REPS)]
    rows = []
    for model in MODELS:
        for rep, rep_seed in enumerate(rep_seeds):
            rng = np.random.default_rng(rep_seed)
            sel = rng.choice(np.arange(len(X_pool)), size=N_BUDGET,
                             replace=False)
            X_sel, y_sel = X_pool[sel], y_pool[sel]
            half = N_BUDGET // 2
            p = rng.permutation(N_BUDGET)
            tr, ca = p[:half], p[half:]
            pred_cal, pred_pop = H.fit_and_predict(
                model, X_sel[tr], y_sel[tr], X_sel[ca], X_pop, rep_seed)
            q = one_sided_quantile(pred_cal - y_sel[ca], 0.90)
            lower = pred_pop - q
            unsafe = y_pop <= TAU
            pred_safe = lower > TAU
            fs = unsafe & pred_safe
            fu = (~unsafe) & (~pred_safe)

            per_eng = (pd.DataFrame({
                "u": u_pop, "fs": fs.astype(float), "fu": fu.astype(float),
                "unsafe": unsafe.astype(float),
                "miss": (fs & unsafe).astype(float)})
                .groupby("u").agg(fs_rate=("fs", "mean"),
                                  fu_rate=("fu", "mean"),
                                  n_unsafe=("unsafe", "sum"),
                                  n_miss=("miss", "sum")))
            fs_r = per_eng["fs_rate"].to_numpy(float)
            fu_r = per_eng["fu_rate"].to_numpy(float)
            nu = per_eng["n_unsafe"].to_numpy(float)
            nm = per_eng["n_miss"].to_numpy(float)
            cond_e = np.divide(nm, nu, out=np.full(len(nu), np.nan),
                               where=nu > 0)
            pop_pfs = float(fs_r.mean())
            pop_pfu = float(fu_r.mean())
            pop_cond = float(np.nanmean(cond_e))
            pop_beta = H.beta_from_p(pop_pfs)
            pop_pass = (pop_beta >= BETA_TARGET and pop_pfu <= PFU_MAX
                        and pop_cond <= M_STAR)

            rng_draw = np.random.default_rng(70_000 + rep_seed)
            n_eng = len(fs_r)
            idx = rng_draw.integers(0, n_eng, size=(N_DRAWS, E_DRAW))
            sub_fs = fs_r[idx]
            bidx = rng_draw.integers(0, E_DRAW, size=(N_DRAWS, N_BOOT, E_DRAW))
            boots = np.take_along_axis(
                sub_fs[:, None, :].repeat(N_BOOT, axis=1), bidx, axis=2
            ).mean(axis=2)
            u95 = np.quantile(boots, 0.95, axis=1)
            beta_cons = np.array([H.beta_from_p(v) for v in u95])
            pfu_sub = fu_r[idx].mean(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                cond_sub = np.nanmean(
                    np.where(nu[idx] > 0, nm[idx] / nu[idx], np.nan), axis=1)
            assessable = (nu[idx] > 0).sum(axis=1) >= 3
            gpass = ((beta_cons >= BETA_TARGET) & (pfu_sub <= PFU_MAX)
                     & (cond_sub <= M_STAR) & assessable)
            agree = max(float(gpass.mean()), 1.0 - float(gpass.mean()))
            rows.append({
                "model": model, "rep": rep,
                "rmse_rul": float(np.sqrt(np.mean((pred_pop - y_pop) ** 2))),
                "q_cycles": q,
                "pop_beta_point": pop_beta, "pop_p_fs": pop_pfs,
                "pop_p_fu": pop_pfu, "pop_cond_miss": pop_cond,
                "pop_guarded_pass": bool(pop_pass),
                "draw_pass_fraction": float(gpass.mean()),
                "draw_assessable": float(assessable.mean()),
                "draw_agreement": agree,
            })
        print(f"[82] {model} done ({time.time() - t0:.0f}s)", flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "cmapss_audit_detail.csv", index=False)
    s = (detail.groupby("model")
         .agg(rmse=("rmse_rul", "median"),
              beta=("pop_beta_point", "median"),
              pfu=("pop_p_fu", "median"),
              cond=("pop_cond_miss", "median"),
              pop_pass=("pop_guarded_pass", lambda x: float(np.mean(x))),
              pass8=("draw_pass_fraction", "mean"),
              assess8=("draw_assessable", "mean"),
              agree8=("draw_agreement", "mean"))
         .reset_index())
    s.to_csv(OUT / "cmapss_audit_summary.csv", index=False)

    lines = ["# R35-B second-domain demonstration: C-MAPSS FD002 turbofan RUL",
             "",
             f"Mapping: engine unit = event; tau = {TAU} cycles; unsafe = "
             f"RUL <= tau; surrogate declares safe iff conformal lower bound "
             f"(level 0.90) > tau. {N_POOL_ENGINES} pool engines, "
             f"{len(pop_units_arr)}-engine population, N = {N_BUDGET} labels, "
             f"{N_REPS} reps, {N_DRAWS} 8-engine draws.", "",
             "| model | RMSE (cycles) | pop beta | pop P_FU | pop cond miss | "
             "pop guarded | 8-engine pass | assessable@8 | agreement@8 |",
             "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |"]
    for _, r in s.iterrows():
        lines.append(
            f"| {r['model']} | {r['rmse']:.1f} | {r['beta']:.2f} | "
            f"{r['pfu']:.3f} | {r['cond']:.3f} | "
            f"{'Pass' if r['pop_pass'] >= 0.5 else 'Fail'} | "
            f"{100 * r['pass8']:.0f}% | {100 * r['assess8']:.0f}% | "
            f"{100 * r['agree8']:.0f}% |")
    text = "\n".join(lines) + "\n"
    (OUT / "R35_B_CMAPSS_REPORT.md").write_text(text, encoding="utf-8")
    (ROUND / "R35_B_CMAPSS_REPORT.md").write_text(text, encoding="utf-8")
    print(f"[82] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
