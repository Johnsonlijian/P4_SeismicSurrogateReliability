"""R35-E4: solver verification and degrading/P-Delta label sensitivity.

(a) Step-halving: re-integrate the 16 protocol systems x 187 disjoint-test
    components at dt = 0.005 (linear-interpolated input) and compare peak IDR
    against the production dt = 0.01 labels.
(b) Cross-solver: scipy solve_ivp (RK45, tight tolerances) on selected pairs
    including threshold-exceeding cases.
(c) Degrading + P-Delta variant: energy-based Bouc-Wen stiffness/strength
    degradation (delta = 0.10) plus story geometric P-Delta, same systems and
    records; checks whether the >= 1% IDR exceedance set and tail ordering
    that the false-safe audit rests on are artifacts of the non-degrading,
    no-P-Delta idealization.
"""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import spearmanr

import r35_harness as H

HERE = Path(__file__).resolve().parent
OUT = H.ROOT / "outputs" / "high_target" / "r35_solver_verification"
ROUND = H.ROOT / "rounds" / "R34_figure_style_normalization_2026-06-10"

_spec = importlib.util.spec_from_file_location("grid22", HERE / "22_nonlinear_mdof_shear_grid.py")
grid22 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(grid22)

DT = 0.01
H_STORY = 3.0
G = 9.81


def protocol_systems() -> list[dict]:
    return [s for s in grid22.system_grid() if s["T1_s"] >= 1.8]


def integrate_variant(gms, dt, n_story, T1, zeta, pattern, yield_drift,
                      alpha_post, p_delta=False, degrade=0.0):
    """22's RK4 Bouc-Wen integrator with optional P-Delta and degradation."""
    n_gm, n_t = gms.shape
    M, C, k_story, w = grid22.build_linear_matrices(n_story, T1, zeta, pattern)
    B = grid22.drift_matrix(n_story)
    Minv = np.linalg.inv(M)
    ones = np.ones(n_story)
    uy = np.full(n_story, yield_drift * H_STORY, dtype=float)
    kg = (G * np.arange(n_story, 0, -1) / H_STORY) if p_delta else np.zeros(n_story)

    u = np.zeros((n_gm, n_story))
    v = np.zeros((n_gm, n_story))
    h = np.zeros((n_gm, n_story))
    eps = np.zeros((n_gm, n_story))
    peak_idr = np.zeros(n_gm)

    def deriv(uu, vv, hh, ee, aa):
        hh = np.clip(hh, -8.0, 8.0)
        drift = uu @ B.T
        drift_v = vv @ B.T
        story_shear = (alpha_post * k_story[None, :] * drift
                       + (1.0 - alpha_post) * k_story[None, :] * uy[None, :] * hh
                       - kg[None, :] * drift)
        restoring = story_shear @ B
        damping = vv @ C.T
        rhs = -damping - restoring - aa[:, None] * ones[None, :]
        acc = rhs @ Minv.T
        abs_h = np.abs(hh)
        nu = 1.0 + degrade * ee
        eta = 1.0 + degrade * ee
        dh = (drift_v
              - nu * (0.5 * np.abs(drift_v) * abs_h * hh
                      + 0.5 * drift_v * abs_h ** 2)) / (uy[None, :] * eta)
        de = ((1.0 - alpha_post) * k_story[None, :] * uy[None, :] * hh * drift_v
              / (k_story[None, :] * uy[None, :] ** 2))
        return vv, np.clip(acc, -1e5, 1e5), np.clip(dh, -1e5, 1e5), de

    for i in range(n_t - 1):
        a0, a1 = gms[:, i], gms[:, i + 1]
        amid = 0.5 * (a0 + a1)
        k1u, k1v, k1h, k1e = deriv(u, v, h, eps, a0)
        k2u, k2v, k2h, k2e = deriv(u + 0.5 * dt * k1u, v + 0.5 * dt * k1v,
                                   h + 0.5 * dt * k1h, eps + 0.5 * dt * k1e, amid)
        k3u, k3v, k3h, k3e = deriv(u + 0.5 * dt * k2u, v + 0.5 * dt * k2v,
                                   h + 0.5 * dt * k2h, eps + 0.5 * dt * k2e, amid)
        k4u, k4v, k4h, k4e = deriv(u + dt * k3u, v + dt * k3v, h + dt * k3h,
                                   eps + dt * k3e, a1)
        u += (dt / 6.0) * (k1u + 2 * k2u + 2 * k3u + k4u)
        v += (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        h += (dt / 6.0) * (k1h + 2 * k2h + 2 * k3h + k4h)
        eps += (dt / 6.0) * (k1e + 2 * k2e + 2 * k3e + k4e)
        u = np.nan_to_num(u, nan=0.0, posinf=1e4, neginf=-1e4)
        v = np.nan_to_num(v, nan=0.0, posinf=1e4, neginf=-1e4)
        h = np.clip(np.nan_to_num(h, nan=0.0, posinf=8.0, neginf=-8.0), -8.0, 8.0)
        eps = np.clip(np.nan_to_num(eps, nan=0.0), 0.0, 50.0)
        idr = np.max(np.abs(u @ B.T) / H_STORY, axis=1)
        peak_idr = np.maximum(peak_idr, idr)
    return peak_idr


def reference_solution(acc, dt, n_story, T1, zeta, pattern, yield_drift,
                       alpha_post):
    """Independent adaptive integrator (solve_ivp RK45) for one record."""
    M, C, k_story, _ = grid22.build_linear_matrices(n_story, T1, zeta, pattern)
    B = grid22.drift_matrix(n_story)
    Minv = np.linalg.inv(M)
    uy = np.full(n_story, yield_drift * H_STORY)
    t_grid = np.arange(len(acc)) * dt

    def rhs(t, y):
        u = y[:n_story]
        v = y[n_story:2 * n_story]
        h = np.clip(y[2 * n_story:], -8.0, 8.0)
        a_g = np.interp(t, t_grid, acc)
        drift = B @ u
        drift_v = B @ v
        shear = alpha_post * k_story * drift + (1 - alpha_post) * k_story * uy * h
        restoring = B.T @ shear
        acc_s = Minv @ (-C @ v - restoring - a_g * np.ones(n_story))
        abs_h = np.abs(h)
        dh = (drift_v - 0.5 * np.abs(drift_v) * abs_h * h
              - 0.5 * drift_v * abs_h ** 2) / uy
        return np.concatenate([v, acc_s, dh])

    sol = solve_ivp(rhs, (0.0, t_grid[-1]), np.zeros(3 * n_story),
                    method="RK45", rtol=1e-8, atol=1e-10,
                    t_eval=t_grid, max_step=dt)
    u = sol.y[:n_story]
    drift = grid22.drift_matrix(n_story) @ u
    return float(np.max(np.abs(drift) / H_STORY))


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    records = H.load_records()
    labels = H.load_labels()
    _, target_events, _, _, _ = H.split_events(records)
    _, test_gm, _, _ = H.disjoint_partition(records, target_events)
    acc_all = np.load(H.REC_DIR / "nsmp_recorded_gm_array.npy")
    gms = acc_all[np.array(sorted(test_gm)), :]
    gm_ids = np.array(sorted(test_gm))
    systems = protocol_systems()
    print(f"[78] {len(systems)} systems x {len(gm_ids)} records", flush=True)

    acc2 = np.repeat(gms, 2, axis=1)[:, :-1]
    acc2[:, 1::2] = 0.5 * (gms[:, :-1] + gms[:, 1:])

    rows = []
    for s_i, s in enumerate(systems):
        args = (s["n_story"], s["T1_s"], s["zeta"], s["pattern"],
                s["yield_drift"], s["alpha_post_yield"])
        idr_base = integrate_variant(gms, DT, *args)
        idr_half = integrate_variant(acc2, DT / 2.0, *args)
        idr_dpd = integrate_variant(gms, DT, *args, p_delta=True, degrade=0.10)
        for j, gm in enumerate(gm_ids):
            rows.append({"system_index": s_i, "gm_id": int(gm), **{k: s[k] for k in
                        ["n_story", "T1_s", "pattern", "yield_drift",
                         "alpha_post_yield"]},
                         "idr_dt01": float(idr_base[j]),
                         "idr_dt005": float(idr_half[j]),
                         "idr_degrade_pdelta": float(idr_dpd[j])})
        print(f"[78] system {s_i + 1}/{len(systems)} ({time.time() - t0:.0f}s)",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "solver_verification_labels.csv", index=False)

    rel = np.abs(df["idr_dt005"] - df["idr_dt01"]) / np.maximum(df["idr_dt01"], 1e-8)
    exceed1 = df["idr_dt01"] > 0.01
    exceed1_half = df["idr_dt005"] > 0.01
    agree1 = float((exceed1 == exceed1_half).mean())
    flip_n = int((exceed1 != exceed1_half).sum())

    big = df[df["idr_dt01"] > 0.005]
    rho_dpd = spearmanr(big["idr_dt01"], big["idr_degrade_pdelta"]).statistic
    ex_dpd = df["idr_degrade_pdelta"] > 0.01
    jacc = float((exceed1 & ex_dpd).sum() / max((exceed1 | ex_dpd).sum(), 1))
    still_unsafe = float(ex_dpd[exceed1].mean()) if exceed1.sum() else float("nan")

    # Cross-solver spot checks: 6 exceeding + 6 random pairs.
    rng = np.random.default_rng(20260617)
    exc_idx = df[exceed1].sample(min(6, int(exceed1.sum())),
                                 random_state=1).index
    rnd_idx = df.sample(6, random_state=2).index
    spot = []
    for idx in list(exc_idx) + list(rnd_idx):
        r = df.loc[idx]
        s = systems[int(r["system_index"])]
        a = acc_all[int(r["gm_id"]), :]
        ref = reference_solution(a, DT, s["n_story"], s["T1_s"], s["zeta"],
                                 s["pattern"], s["yield_drift"],
                                 s["alpha_post_yield"])
        spot.append({"gm_id": int(r["gm_id"]), "system_index": int(r["system_index"]),
                     "idr_dt01": float(r["idr_dt01"]), "idr_ref_rk45": ref,
                     "rel_diff": abs(ref - r["idr_dt01"]) / max(r["idr_dt01"], 1e-8)})
        print(f"[78] spot {len(spot)}/12 ({time.time() - t0:.0f}s)", flush=True)
    spot_df = pd.DataFrame(spot)
    spot_df.to_csv(OUT / "cross_solver_spot_checks.csv", index=False)

    lines = ["# R35-E4 solver verification and degrading/P-Delta sensitivity", "",
             f"Scope: 16 protocol systems x {len(gm_ids)} disjoint-test records "
             f"({len(df)} pairs).", "",
             "## Step-halving (dt 0.01 -> 0.005)", "",
             f"- Median relative peak-IDR difference: {float(rel.median()):.2e}",
             f"- p95: {float(rel.quantile(0.95)):.2e}; max: {float(rel.max()):.2e}",
             f"- 1% IDR exceedance classification agreement: {100 * agree1:.2f}% "
             f"({flip_n} flips of {len(df)})", "",
             "## Cross-solver spot checks (RK45, rtol 1e-8)", "",
             f"- Median relative difference: {float(spot_df['rel_diff'].median()):.2e}",
             f"- Max: {float(spot_df['rel_diff'].max()):.2e} over {len(spot_df)} pairs", "",
             "## Degrading (delta=0.10) + P-Delta variant", "",
             f"- Spearman rho of peak IDR (pairs with IDR > 0.5%): {rho_dpd:.3f}",
             f"- 1% exceedance Jaccard overlap: {jacc:.3f}",
             f"- Fraction of baseline-unsafe pairs still unsafe: {100 * still_unsafe:.1f}%",
             f"- Baseline unsafe pairs: {int(exceed1.sum())}; variant unsafe: {int(ex_dpd.sum())}", ""]
    text = "\n".join(lines)
    (OUT / "R35_E4_SOLVER_VERIFICATION_REPORT.md").write_text(text, encoding="utf-8")
    (ROUND / "R35_E4_SOLVER_VERIFICATION_REPORT.md").write_text(text, encoding="utf-8")
    print(f"[78] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
