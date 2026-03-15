import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def evaluate_policy(
    df: pd.DataFrame,
    flag_thr: float,
    block_thr: float,
    block_min_margin: float,
    safe_recovery_max_attack_sim: float,
    safe_recovery_min_benign_sim: float,
    safe_recovery_max_margin: float,
):
    attack = df["layer_b_attack_similarity"].to_numpy()
    benign = df["layer_b_benign_similarity"].to_numpy()
    margin = df["layer_b_margin"].to_numpy()
    labels = df["true_label"].to_numpy()

    verdict = np.full(len(df), "allow", dtype=object)

    block_mask = (attack >= block_thr) & (margin >= block_min_margin)
    verdict[block_mask] = "block"

    flag_band = (attack >= flag_thr) & ~block_mask
    recovery_allow = (
        flag_band
        & (attack <= safe_recovery_max_attack_sim)
        & (benign >= safe_recovery_min_benign_sim)
        & (margin <= safe_recovery_max_margin)
    )
    verdict[flag_band] = "flag"
    verdict[recovery_allow] = "allow"

    safe = labels == 0
    mal = labels == 1

    safe_allow = np.sum((verdict == "allow") & safe)
    safe_flag = np.sum((verdict == "flag") & safe)
    safe_block = np.sum((verdict == "block") & safe)
    mal_allow = np.sum((verdict == "allow") & mal)
    mal_flag = np.sum((verdict == "flag") & mal)
    mal_block = np.sum((verdict == "block") & mal)

    safe_total = np.sum(safe)
    mal_total = np.sum(mal)
    total = len(df)
    block_total = safe_block + mal_block

    metrics = {
        "flag_thr": flag_thr,
        "block_thr": block_thr,
        "block_min_margin": block_min_margin,
        "safe_recovery_max_attack_sim": safe_recovery_max_attack_sim,
        "safe_recovery_min_benign_sim": safe_recovery_min_benign_sim,
        "safe_recovery_max_margin": safe_recovery_max_margin,
        "safe_allow": int(safe_allow),
        "safe_flag": int(safe_flag),
        "safe_block": int(safe_block),
        "mal_allow": int(mal_allow),
        "mal_flag": int(mal_flag),
        "mal_block": int(mal_block),
        "safe_allow_rate": safe_allow / safe_total if safe_total else 0.0,
        "safe_block_rate": safe_block / safe_total if safe_total else 0.0,
        "mal_allow_rate": mal_allow / mal_total if mal_total else 0.0,
        "mal_block_rate": mal_block / mal_total if mal_total else 0.0,
        "flag_rate": (safe_flag + mal_flag) / total if total else 0.0,
        "block_precision": mal_block / block_total if block_total else 0.0,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Constrained threshold sweep for Layer B")
    parser.add_argument("--input", required=True, help="CSV from tests/layer_b/test_layer_b.py")
    parser.add_argument("--max-mal-allow", type=float, default=0.005, help="Max malicious allow rate")
    parser.add_argument("--min-block-precision", type=float, default=0.90, help="Min block precision")
    parser.add_argument("--top", type=int, default=20, help="How many best rows to print")
    parser.add_argument("--output", default="test_results/ablation/layer_b_threshold_sweep.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = {
        "true_label",
        "layer_b_attack_similarity",
        "layer_b_benign_similarity",
        "layer_b_margin",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    rows = []

    flag_grid = np.arange(0.50, 0.71, 0.02)
    block_grid = np.arange(0.72, 0.86, 0.02)
    margin_grid = np.arange(0.00, 0.081, 0.02)
    recover_attack_grid = np.arange(0.60, 0.73, 0.03)
    recover_benign_grid = np.arange(0.60, 0.81, 0.05)
    recover_margin_grid = np.arange(-0.10, -0.019, 0.02)

    for flag_thr in flag_grid:
        for block_thr in block_grid:
            if block_thr <= flag_thr:
                continue
            for block_min_margin in margin_grid:
                for rec_attack in recover_attack_grid:
                    for rec_benign in recover_benign_grid:
                        for rec_margin in recover_margin_grid:
                            rows.append(
                                evaluate_policy(
                                    df=df,
                                    flag_thr=float(flag_thr),
                                    block_thr=float(block_thr),
                                    block_min_margin=float(block_min_margin),
                                    safe_recovery_max_attack_sim=float(rec_attack),
                                    safe_recovery_min_benign_sim=float(rec_benign),
                                    safe_recovery_max_margin=float(rec_margin),
                                )
                            )

    out = pd.DataFrame(rows)

    feasible = out[
        (out["mal_allow_rate"] <= args.max_mal_allow)
        & (out["block_precision"] >= args.min_block_precision)
    ].copy()

    if feasible.empty:
        print("No feasible configuration met the constraints.")
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"Saved full sweep to {out_path}")
        return

    feasible = feasible.sort_values(
        by=["safe_block_rate", "safe_allow_rate", "flag_rate"],
        ascending=[True, False, True],
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feasible.to_csv(out_path, index=False)

    print("Best feasible candidates:")
    print(
        feasible.head(args.top)[
            [
                "flag_thr",
                "block_thr",
                "block_min_margin",
                "safe_recovery_max_attack_sim",
                "safe_recovery_min_benign_sim",
                "safe_recovery_max_margin",
                "safe_allow_rate",
                "safe_block_rate",
                "mal_allow_rate",
                "block_precision",
                "flag_rate",
            ]
        ].to_string(index=False)
    )
    print(f"Saved feasible sweep to {out_path}")


if __name__ == "__main__":
    main()
