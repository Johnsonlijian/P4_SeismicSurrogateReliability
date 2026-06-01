"""R28 true event-disjoint large-budget prediction export.

This script reruns the event-disjoint target calibration/test protocol for
larger nominal target-label budgets instead of extrapolating from the R27 trace.
It writes full calibration and test residual predictions for focused models so
false-safe reliability can be recomputed at N=1000 and N=2000.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REC = ROOT / "outputs" / "high_target" / "recorded_nsmp_full"
MDOF = ROOT / "outputs" / "high_target" / "nonlinear_mdof_grid_full"
OUT = ROOT / "outputs" / "high_target" / "r28_event_disjoint_large_budget"


def import_script(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, HERE / file_name)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


mdof_mod = import_script("r28_mdof", "23_nonlinear_mdof_label_budget.py")
trace_mod = import_script("r28_trace", "52_residual_trace_mechanism.py")
train_mod = mdof_mod.train_mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R28 event-disjoint large-budget export")
    p.add_argument("--n-reps", type=int, default=10)
    p.add_argument("--budgets", default="50,100,250,500,1000,2000")
    p.add_argument("--models", default="ridge_direct,lgbm_direct,xgb_direct,scratch_mlp")
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--target-event-fraction", type=float, default=0.35)
    p.add_argument("--sample-per-cell", type=int, default=5000, help="Use >= test rows for full test prediction export.")
    return p.parse_args()


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_text(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def target_event_disjoint_split(records: pd.DataFrame, target_events: list[str]) -> tuple[list[int], list[int], list[str], list[str]]:
    target_events_arr = np.array(sorted(target_events))
    rng = np.random.default_rng(20260613)
    perm = rng.permutation(target_events_arr)
    cut = max(1, len(perm) // 2)
    fit_events = sorted(perm[:cut].tolist())
    test_events = sorted(perm[cut:].tolist())
    fit_gm = sorted(records[records["event_id"].isin(fit_events)]["gm_id"].to_numpy(int).tolist())
    test_gm = sorted(records[records["event_id"].isin(test_events)]["gm_id"].to_numpy(int).tolist())
    return fit_gm, test_gm, fit_events, test_events


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    mdof_mod.REC = REC
    mdof_mod.MDOF = MDOF

    records = pd.read_csv(REC / "nsmp_recorded_records.csv")
    rec_labels = mdof_mod.load_labels("recorded")
    source_events, target_events, source_gm, _pool_gm, _test_gm = mdof_mod.split_events(records, args.target_event_fraction)
    fit_gm, test_gm, fit_events, test_events = target_event_disjoint_split(records, target_events)

    pool_df, X_pool, y_pool = mdof_mod.merge_features(rec_labels, records, "log_max_idr", gm_ids=fit_gm, source_period="target")
    test_df, X_test, y_test = mdof_mod.merge_features(rec_labels, records, "log_max_idr", gm_ids=test_gm, source_period="target")
    pool_df = pool_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    exceed_mask = np.asarray(y_test > np.log10(0.01), dtype=bool)

    _, X_rec_pre, y_rec_pre = mdof_mod.merge_features(rec_labels, records, "log_max_idr", gm_ids=source_gm, source_period="seen")
    _, X_rec_val, y_rec_val = mdof_mod.merge_features(rec_labels, records, "log_max_idr", gm_ids=fit_gm, source_period="seen")
    foundation, sc, _ = train_mod.fit_foundation(X_rec_pre, y_rec_pre, X_rec_val, y_rec_val, seed=20260613)

    budgets = parse_ints(args.budgets)
    models = parse_text(args.models)
    feasible_budgets = [n for n in budgets if n <= len(X_pool)]
    skipped_budgets = [n for n in budgets if n > len(X_pool)]
    rows, samples = trace_mod.run_family(
        family="recorded_mdof_event_disjoint_target_conformal_r28_large_budget",
        foundation=foundation,
        sc=sc,
        pool_df=pool_df,
        X_pool=X_pool,
        y_pool=y_pool,
        test_df=test_df,
        X_test=X_test,
        y_test=y_test,
        exceed_mask=exceed_mask,
        budgets=feasible_budgets,
        models=models,
        alpha=args.alpha,
        n_reps=args.n_reps,
        sample_per_cell=args.sample_per_cell,
    )
    summary = pd.DataFrame(rows)
    samples_df = pd.DataFrame(samples)
    agg = trace_mod.aggregate(summary)
    summary.to_csv(OUT / "event_disjoint_large_budget_summary.csv", index=False)
    agg.to_csv(OUT / "event_disjoint_large_budget_aggregate.csv", index=False)
    samples_df.to_csv(OUT / "event_disjoint_large_budget_predictions.csv", index=False)

    meta = {
        "protocol": "recorded_mdof_event_disjoint_target_conformal_r28_large_budget",
        "source_event_count": len(source_events),
        "target_event_count": len(target_events),
        "target_fit_event_count": len(fit_events),
        "target_test_event_count": len(test_events),
        "target_fit_events": fit_events,
        "target_test_events": test_events,
        "target_fit_components": len(fit_gm),
        "target_test_components": len(test_gm),
        "target_fit_rows": int(len(X_pool)),
        "target_test_rows": int(len(X_test)),
        "target_test_rows_above_1pct": int(exceed_mask.sum()),
        "requested_budgets": budgets,
        "feasible_budgets": feasible_budgets,
        "skipped_budgets": skipped_budgets,
        "models": models,
        "n_reps": args.n_reps,
        "alpha": args.alpha,
        "sample_per_cell": args.sample_per_cell,
        "export_note": "sample_per_cell is set above the calibration/test split sizes, so calibration and test rows are exported fully for feasible budgets.",
    }
    (OUT / "event_disjoint_large_budget_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lines = [
        "# R28 event-disjoint large-budget export",
        "",
        "This run regenerates prediction residuals for event-disjoint target calibration/test splits rather than extrapolating from the N=500 trace.",
        "",
        f"- Fit/calibration event rows available: {len(X_pool)}",
        f"- Test event rows: {len(X_test)}",
        f"- Feasible nominal target-label budgets: {', '.join(str(n) for n in feasible_budgets)}",
        f"- Skipped requested budgets: {', '.join(str(n) for n in skipped_budgets) if skipped_budgets else 'none'}",
        f"- Models: {', '.join(models)}",
        f"- Repetitions: {args.n_reps}",
        "",
        "## Output files",
        "",
        "- `event_disjoint_large_budget_predictions.csv`: full calibration and test residual predictions.",
        "- `event_disjoint_large_budget_summary.csv`: per-run residual and conformal metrics.",
        "- `event_disjoint_large_budget_aggregate.csv`: model-budget aggregates.",
    ]
    (OUT / "R28_EVENT_DISJOINT_LARGE_BUDGET_EXPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[66] wrote R28 event-disjoint large-budget outputs to {OUT}", flush=True)


if __name__ == "__main__":
    main()
