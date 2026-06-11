"""R35-E7: how much event evidence does a screening verdict need?

Turns the paper's headline ("safety evidence is counted in damaging events")
into a quantitative design rule. Treating the 37 non-fit events as the
archive population (models fitted/calibrated on the original 7 fit events),
we draw test sets of E events WITH replacement (superpopulation analogue)
for E = 6..32 and measure, per surrogate and budget:

- empirical coverage of the event-bootstrap U95 bound vs E;
- the guarded-verdict pass fraction vs E (verdict stabilization curve);
- the fraction of draws that are assessable (>= 3 unsafe-bearing events);
- the population-limit verdict (beta, P_FU, conditional miss at all 37 events).

Everything reuses per-event rates: 24 model fits, then pure numpy.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import r35_harness as H

OUT = H.ROOT / "outputs" / "high_target" / "r35_evidence_scaling"
ROUND = H.ROOT / "rounds" / "R34_figure_style_normalization_2026-06-10"

MODELS = ["ridge_direct", "lgbm_direct", "xgb_direct", "scratch_mlp"]
BUDGETS = [500, 2000]
N_REPS = 3
E_GRID = [6, 8, 12, 16, 20, 24, 28, 32]
N_DRAWS = 400
N_BOOT = 1000
BETA_TARGET = 2.5
PFU_MAX = 0.30
M_STAR = 0.25
E_MIN_UNSAFE = 3


def per_event_tables(records, labels, fit_gm, fit_events):
    pop_records = records[~records["event_id"].isin(fit_events)]
    pop_gm = sorted(pop_records["gm_id"].to_numpy(int).tolist())
    pop_df, X_popu, y_popu = H.merge_features(labels, records, gm_ids=pop_gm,
                                              source_period="target")
    return pop_df, X_popu, y_popu


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    records = H.load_records()
    labels = H.load_labels()
    _, target_events, _, _, _ = H.split_events(records)
    fit_gm, _, fit_events, _ = H.disjoint_partition(records, target_events)
    _, X_pool, y_pool = H.merge_features(labels, records, gm_ids=fit_gm,
                                         source_period="target")
    pop_df, X_popu, y_popu = per_event_tables(records, labels, fit_gm, fit_events)
    ev_pop = pop_df["event_id"].to_numpy()
    events_unique = np.array(sorted(pd.unique(ev_pop)))
    n_ev = len(events_unique)
    print(f"[80] population {n_ev} events, {len(X_popu)} rows", flush=True)

    rep_seeds = [int(s) for s in
                 np.random.default_rng(20260618).integers(1, 10 ** 8, N_REPS)]
    curve_rows, pop_rows = [], []
    for model in MODELS:
        for N in BUDGETS:
            for rep, rep_seed in enumerate(rep_seeds):
                rng = np.random.default_rng(rep_seed)
                sel = rng.choice(np.arange(len(X_pool)), size=N, replace=False)
                X_sel, y_sel = X_pool[sel], y_pool[sel]
                half = N // 2
                perm = rng.permutation(N)
                tr, ca = perm[:half], perm[half:]
                pred_calib, pred_pop = H.fit_and_predict(
                    model, X_sel[tr], y_sel[tr], X_sel[ca], X_popu, rep_seed)
                q = H.conformal_quantile(np.abs(y_sel[ca] - pred_calib), 0.10)
                upper = pred_pop + q
                unsafe = y_popu > H.LOG1PCT
                fs = unsafe & (upper <= H.LOG1PCT)
                fu = (~unsafe) & (upper > H.LOG1PCT)
                df = pd.DataFrame({
                    "e": ev_pop, "fs": fs.astype(float), "fu": fu.astype(float),
                    "unsafe": unsafe.astype(float),
                    "miss": (fs & unsafe).astype(float)})
                per_ev = df.groupby("e").agg(
                    fs_rate=("fs", "mean"), fu_rate=("fu", "mean"),
                    n_unsafe=("unsafe", "sum"), n_miss=("miss", "sum"),
                ).reindex(events_unique)
                fs_r = per_ev["fs_rate"].to_numpy(float)
                fu_r = per_ev["fu_rate"].to_numpy(float)
                n_unsafe_e = per_ev["n_unsafe"].to_numpy(float)
                n_miss_e = per_ev["n_miss"].to_numpy(float)
                cond_e = np.divide(n_miss_e, n_unsafe_e,
                                   out=np.full(n_ev, np.nan),
                                   where=n_unsafe_e > 0)

                pop_pfs = float(fs_r.mean())
                pop_pfu = float(fu_r.mean())
                pop_cond = float(np.nanmean(cond_e))
                pop_beta = H.beta_from_p(pop_pfs)
                pop_pass = (pop_beta >= BETA_TARGET and pop_pfu <= PFU_MAX
                            and pop_cond <= M_STAR)
                pop_rows.append({"model": model, "N": N, "rep": rep,
                                 "pop_p_fs": pop_pfs, "pop_beta_point": pop_beta,
                                 "pop_p_fu": pop_pfu, "pop_cond_miss": pop_cond,
                                 "pop_guarded_pass": bool(pop_pass),
                                 "pop_unsafe_events": int((n_unsafe_e > 0).sum())})

                rng_draw = np.random.default_rng(60_000 + rep_seed)
                for E in E_GRID:
                    idx = rng_draw.integers(0, n_ev, size=(N_DRAWS, E))
                    sub_fs = fs_r[idx]                      # draws x E
                    sub_mean = sub_fs.mean(axis=1)
                    # bootstrap U95 per draw, vectorized.
                    bidx = rng_draw.integers(0, E, size=(N_DRAWS, N_BOOT, E))
                    boots = np.take_along_axis(
                        sub_fs[:, None, :].repeat(N_BOOT, axis=1), bidx, axis=2
                    ).mean(axis=2)
                    u95 = np.quantile(boots, 0.95, axis=1)
                    beta_cons = np.array([H.beta_from_p(p) for p in u95])
                    pfu_sub = fu_r[idx].mean(axis=1)
                    nu_sub = n_unsafe_e[idx]
                    nm_sub = n_miss_e[idx]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        cond_sub_e = np.where(nu_sub > 0, nm_sub / nu_sub, np.nan)
                    cond_sub = np.nanmean(cond_sub_e, axis=1)
                    n_unsafe_events = (nu_sub > 0).sum(axis=1)
                    assessable = n_unsafe_events >= E_MIN_UNSAFE
                    gpass = ((beta_cons >= BETA_TARGET) & (pfu_sub <= PFU_MAX)
                             & (cond_sub <= M_STAR) & assessable)
                    curve_rows.append({
                        "model": model, "N": N, "rep": rep, "E": E,
                        "coverage_u95": float((u95 >= pop_pfs).mean()),
                        "pass_fraction": float(gpass.mean()),
                        "assessable_fraction": float(assessable.mean()),
                        "verdict_agreement": float(max(gpass.mean(),
                                                       1.0 - gpass.mean())),
                    })
            print(f"[80] {model} N={N} ({time.time() - t0:.0f}s)", flush=True)

    curves = pd.DataFrame(curve_rows)
    curves.to_csv(OUT / "evidence_scaling_detail.csv", index=False)
    pop = pd.DataFrame(pop_rows)
    pop.to_csv(OUT / "population_limit_verdicts.csv", index=False)

    cs = (curves.groupby(["model", "N", "E"])
          .agg(coverage_u95=("coverage_u95", "mean"),
               pass_fraction=("pass_fraction", "mean"),
               assessable=("assessable_fraction", "mean"),
               agreement=("verdict_agreement", "mean"))
          .reset_index())
    cs.to_csv(OUT / "evidence_scaling_summary.csv", index=False)
    ps_ = (pop.groupby(["model", "N"])
           .agg(pop_beta_point=("pop_beta_point", "median"),
                pop_p_fu=("pop_p_fu", "median"),
                pop_cond_miss=("pop_cond_miss", "median"),
                pass_share=("pop_guarded_pass", "mean"),
                unsafe_events=("pop_unsafe_events", "median"))
           .reset_index())

    lines = ["# R35-E7 evidence-scaling study", "",
             f"Population = {n_ev} non-fit events; draws with replacement; "
             f"guarded verdict = beta>=2.5 & P_FU<=0.30 & cond<=0.25 & "
             f">= {E_MIN_UNSAFE} unsafe-bearing events.", "",
             "## Population-limit (all 37 events) verdicts", "",
             "| model | N | beta (point) | P_FU | cond. miss | guarded pass | unsafe-bearing events |",
             "| --- | ---: | ---: | ---: | ---: | --- | ---: |"]
    for _, r in ps_.iterrows():
        lines.append(f"| {r['model']} | {int(r['N'])} | "
                     f"{r['pop_beta_point']:.2f} | {r['pop_p_fu']:.3f} | "
                     f"{r['pop_cond_miss']:.3f} | "
                     f"{'Pass' if r['pass_share'] >= 0.5 else 'Fail'} | "
                     f"{int(r['unsafe_events'])} |")
    lines += ["", "## Verdict stabilization vs number of test events E "
              "(N=2000, mean over reps)", "",
              "| model | E | U95 coverage | pass fraction | assessable | agreement |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for _, r in cs[cs["N"].eq(2000)].iterrows():
        lines.append(f"| {r['model']} | {int(r['E'])} | "
                     f"{100 * r['coverage_u95']:.0f}% | "
                     f"{100 * r['pass_fraction']:.0f}% | "
                     f"{100 * r['assessable']:.0f}% | "
                     f"{100 * r['agreement']:.0f}% |")
    text = "\n".join(lines) + "\n"
    (OUT / "R35_E7_EVIDENCE_SCALING_REPORT.md").write_text(text, encoding="utf-8")
    (ROUND / "R35_E7_EVIDENCE_SCALING_REPORT.md").write_text(text, encoding="utf-8")
    print(f"[80] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
