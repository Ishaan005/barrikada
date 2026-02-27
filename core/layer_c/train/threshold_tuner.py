from core.settings import Settings

import numpy as np
import logging

log = logging.getLogger(__name__)

_settings = Settings()

def tune_routing_thresholds(
    y_true,
    scores,
    target_block_precision: float | None = None,
    max_malicious_allow_rate: float | None = None,
    max_safe_fpr: float | None = None,
    min_flag_band: float | None = None,
):
    cfg = _settings
    if target_block_precision is None:
        target_block_precision = cfg.layer_c_tune_target_block_precision
    if max_malicious_allow_rate is None:
        max_malicious_allow_rate = cfg.layer_c_tune_max_malicious_allow_rate
    if max_safe_fpr is None:
        max_safe_fpr = cfg.layer_c_tune_max_safe_fpr
    if min_flag_band is None:
        min_flag_band = cfg.layer_c_tune_min_flag_band

    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores)

    steps = cfg.layer_c_tune_grid_steps
    low_grid = np.linspace(cfg.layer_c_tune_low_grid_start, cfg.layer_c_tune_low_grid_end, steps)
    high_grid = np.linspace(cfg.layer_c_tune_high_grid_start, cfg.layer_c_tune_high_grid_end, steps)

    # Pre-compute label masks for vectorised scoring
    is_mal = y == 1
    is_safe = ~is_mal
    mal_total = max(1, int(np.sum(is_mal)))
    safe_total = max(1, int(np.sum(is_safe)))

    best = None

    # Diagnostic counters to identify which constraint is the bottleneck
    n_checked = 0
    n_fail_band = 0
    n_fail_block_prec = 0
    n_fail_mal_allow = 0
    n_fail_safe_fpr = 0

    for low in low_grid:
        for high in high_grid:
            if low >= high or (high - low) < min_flag_band:
                n_fail_band += 1
                continue
            n_checked += 1

            # Three-way partition of traffic
            pred_allow = s < low
            pred_block = s >= high
            pred_flag = (~pred_allow) & (~pred_block)

            # Check block precision
            tp_block = np.sum(pred_block & is_mal)
            fp_block = np.sum(pred_block & is_safe)
            block_precision = (tp_block / (tp_block + fp_block)) if (tp_block + fp_block) else 1.0

            if block_precision < target_block_precision:
                n_fail_block_prec += 1
                continue

            # Don't allow too many malicious through
            mal_allow_rate = np.sum(pred_allow & is_mal) / mal_total
            if mal_allow_rate > max_malicious_allow_rate:
                n_fail_mal_allow += 1
                continue

            # Limit false-positive rate on safe texts (flag + block)
            safe_fpr = float(np.sum((pred_flag | pred_block) & is_safe) / safe_total)
            if safe_fpr > max_safe_fpr:
                n_fail_safe_fpr += 1
                continue

            flag_rate = float(np.mean(pred_flag))
            block_rate = float(np.mean(pred_block))
            allow_rate = float(np.mean(pred_allow))
            block_recall = float(tp_block / mal_total)

            # Objective: maximise block_recall while penalising FN (mal→allow) and FP (safe→flag/block)
            # Deliberately does NOT reward allow_rate — that was causing low to be pushed as high as the
            # mal_allow_rate constraint permits, letting through the maximum allowed malicious traffic.
            score = (
                cfg.layer_c_tune_w_block_recall * block_recall
                - cfg.layer_c_tune_w_mal_allow_penalty * mal_allow_rate
                - cfg.layer_c_tune_w_safe_fpr_penalty * safe_fpr
            )

            if best is None or score > best[0]:
                best = (
                    score,
                    {
                        "low": float(low),
                        "high": float(high),
                        "val_flag_rate": flag_rate,
                        "val_block_rate": block_rate,
                        "val_allow_rate": allow_rate,
                        "val_safe_fpr": safe_fpr,
                        "val_block_precision": float(block_precision),
                        "val_block_recall": block_recall,
                        "val_malicious_allow_rate": float(mal_allow_rate),
                    },
                )

    # Always print constraint diagnostics so failures are visible
    print(
        f"[tune] grid checked={n_checked}  "
        f"fail_block_prec(>={target_block_precision:.2f})={n_fail_block_prec}  "
        f"fail_mal_allow(<={max_malicious_allow_rate:.2f})={n_fail_mal_allow}  "
        f"fail_safe_fpr(<={max_safe_fpr:.2f})={n_fail_safe_fpr}  "
        f"{'-> FOUND solution' if best is not None else '-> NO solution — falling back to defaults'}"
    )

    #to return a default if no thresholds found
    if best is None:
        log.warning(
            "tune_routing_thresholds: no (low, high) pair satisfied all constraints "
            "[block_prec>=%.2f  mal_allow<=%.2f  safe_fpr<=%.2f]. "
            "Rejection counts — block_prec: %d  mal_allow: %d  safe_fpr: %d. "
            "Falling back to defaults (low=%.2f, high=%.2f). Relax the binding constraint above.",
            target_block_precision, max_malicious_allow_rate, max_safe_fpr,
            n_fail_block_prec, n_fail_mal_allow, n_fail_safe_fpr,
            _settings.layer_c_low_threshold, _settings.layer_c_high_threshold,
        )
        return {
            "low": _settings.layer_c_low_threshold,
            "high": _settings.layer_c_high_threshold,
            "val_flag_rate": 0.0,
            "val_block_rate": 0.0,
            "val_allow_rate": 0.0,
            "val_safe_fpr": None,
            "val_block_precision": 0.0,
            "val_block_recall": 0.0,
            "val_malicious_allow_rate": 0.0,
        }

    return best[1]