import math
from typing import Dict, Tuple, List

import numpy as np

from scoring import score_row, sweet_spot, zscore, geo_penalty


def _safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        return float(str(x).replace(",", ""))
    except Exception:
        return default


def component_scores(r: Dict, weights: Dict) -> Dict[str, float]:
    prem = _safe_float(r.get("total_premium"))
    win = r.get("winnability")
    year = r.get("building_year")
    good_constr = 1.0 if r.get("is_good_construction") else 0.0
    tiv = _safe_float(r.get("tiv"))
    fresh = r.get("fresh_days")

    s_prem = sweet_spot(
        prem,
        weights.get("premium_lo", 0),
        weights.get("premium_mid_lo", 0),
        weights.get("premium_mid_hi", 0),
        weights.get("premium_hi", 0),
    )
    s_win = 0.0 if win is None or (isinstance(win, float) and np.isnan(win)) else float(win)
    s_year = 0.0 if year is None or (isinstance(year, float) and np.isnan(year)) else zscore(
        year, 1990, 2022
    )
    s_con = good_constr
    s_tiv = 1.0 - zscore(tiv, 0, weights.get("tiv_hi", 150_000_000)) if not np.isnan(tiv) else 0.0
    s_fresh = (
        1.0 - zscore(fresh, 0, 90)
        if fresh is not None and not (isinstance(fresh, float) and np.isnan(fresh))
        else 0.5
    )

    return {
        "s_prem": float(s_prem),
        "s_win": float(s_win),
        "s_year": float(s_year),
        "s_con": float(s_con),
        "s_tiv": float(s_tiv),
        "s_fresh": float(s_fresh),
    }


def reasons_for_filters(r: Dict, filters: Dict) -> List[str]:
    reasons: List[str] = []
    lob = (r.get("line_of_business") or "").upper()
    if "lob_in" in filters:
        allowed = [x.upper() for x in filters["lob_in"]]
        if lob not in allowed:
            reasons.append("Line of business not in allowed list")

    if filters.get("new_business_only"):
        sub_type = (r.get("renewal_or_new_business") or "").upper()
        if sub_type != "NEW_BUSINESS":
            reasons.append("Submission is not NEW_BUSINESS")

    if "loss_ratio_max" in filters:
        lr = r.get("loss_ratio")
        if lr is None or (isinstance(lr, float) and np.isnan(lr)):
            # If we require a max loss ratio, missing value should be flagged
            reasons.append("Missing loss ratio for evaluation")
        elif float(lr) > float(filters["loss_ratio_max"]):
            reasons.append(f"Loss ratio {float(lr):.2f} exceeds max {filters['loss_ratio_max']:.2f}")

    if "tiv_max" in filters:
        tiv = _safe_float(r.get("tiv"))
        if not np.isnan(tiv) and tiv > float(filters["tiv_max"]):
            reasons.append(f"TIV ${tiv:,.0f} exceeds max ${float(filters['tiv_max']):,.0f}")

    if "premium_range" in filters:
        lo, hi = filters["premium_range"]
        prem = _safe_float(r.get("total_premium"))
        if np.isnan(prem) or prem < float(lo) or prem > float(hi):
            reasons.append(
                f"Premium outside range ${float(lo):,.0f}-${float(hi):,.0f}"
            )

    if "min_winnability" in filters:
        win = r.get("winnability")
        if win is None or (isinstance(win, float) and np.isnan(win)):
            reasons.append("Missing winnability for evaluation")
        elif float(win) < float(filters["min_winnability"]):
            reasons.append(
                f"Winnability {float(win):.2f} below min {float(filters['min_winnability']):.2f}"
            )

    if "min_year" in filters:
        yr = r.get("building_year")
        if yr is None or (isinstance(yr, float) and np.isnan(yr)):
            # allow missing as acceptable only when rule intends to require a minimum
            reasons.append("Missing building year for evaluation")
        elif int(yr) < int(filters["min_year"]):
            reasons.append(f"Oldest building {int(yr)} before {int(filters['min_year'])}")

    if filters.get("good_construction_only"):
        if not r.get("is_good_construction"):
            reasons.append("Construction type not in acceptable set")

    return reasons


def classify_for_mode_row(r: Dict, weights: Dict, filters: Dict, state_counts: Dict) -> Tuple[str, List[str], float, float]:
    """Return (status, reasons, mode_score, priority_score)."""
    reasons = reasons_for_filters(r, filters)

    # Score with geo concentration penalty baked in via score_row
    s = float(score_row(r, weights, state_counts))

    # Determine status
    if len(reasons) > 0:
        status = "OUT"
    else:
        comps = component_scores(r, weights)
        # target if very strong score or strong components
        if s >= 0.80 or (
            comps["s_prem"] >= 0.9
            and comps["s_win"] >= 0.6
            and comps["s_con"] >= 1.0
            and comps["s_year"] >= 0.6
            and comps["s_tiv"] >= 0.6
        ):
            status = "TARGET"
        else:
            status = "IN"

    # Priority mirrors score here; could blend in explicit win weight again if desired
    priority_score = s
    return status, reasons, s, priority_score

