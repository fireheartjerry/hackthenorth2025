import math


def _num(x, default=None):
    try:
        if x is None:
            return default
        return float(str(x).replace(",", "").strip())
    except Exception:
        return default


def _normalize_weights(weights: dict) -> dict:
    w = dict(weights or {})
    core_keys = ["w_prem", "w_win", "w_year", "w_con", "w_tiv", "w_fresh"]
    # clamp core weights to [0,1]
    total = 0.0
    for k in core_keys:
        v = _num(w.get(k), 0.0) or 0.0
        v = max(0.0, min(1.0, float(v)))
        w[k] = v
        total += v
    if total <= 0:
        for k in core_keys:
            w[k] = 1.0 / len(core_keys)
    else:
        for k in core_keys:
            w[k] = w[k] / total

    # sweet-spot guardrails
    plo = _num(w.get("premium_lo"), 100_000) or 100_000
    phi = _num(w.get("premium_hi"), 1_500_000) or 1_500_000
    if phi < plo:
        plo, phi = phi, plo
    span = max(1.0, phi - plo)
    pml = _num(w.get("premium_mid_lo"), plo + 0.30 * span)
    pmh = _num(w.get("premium_mid_hi"), plo + 0.70 * span)
    ordered = sorted([plo, pml, pmh, phi])
    w["premium_lo"], w["premium_mid_lo"], w["premium_mid_hi"], w["premium_hi"] = [float(x) for x in ordered]

    # tiv bound
    tiv_hi = _num(w.get("tiv_hi"), max(ordered[-1] * 10.0, 1_000_000.0)) or max(ordered[-1] * 10.0, 1_000_000.0)
    w["tiv_hi"] = float(max(1_000_000.0, tiv_hi))
    return w


def seeds_from_answers(answers: dict) -> dict:
    """
    Deterministically map questionnaire answers to initial rules.
    Returns {filters:{...}, weights:{...}}
    """
    a = answers or {}
    aggr = int(a.get("aggressiveness", 3) or 3)
    aggr = max(1, min(5, aggr))
    objective = (a.get("objective") or "balance").strip().lower()
    prem_lo = _num(a.get("premium_lo"), 100_000) or 100_000
    prem_hi = _num(a.get("premium_hi"), 1_500_000) or 1_500_000
    if prem_hi < prem_lo:
        prem_lo, prem_hi = prem_hi, prem_lo
    max_lr = _num(a.get("max_lr"), 0.65) or 0.65
    tiv_max = _num(a.get("tiv_max"), 150_000_000) or 150_000_000
    new_only = bool(a.get("new_only", True))
    strict_construction = bool(a.get("strict_construction", False))

    # maps by aggressiveness
    lr_map = {1: 0.50, 2: 0.60, 3: 0.70, 4: 0.80, 5: 0.90}
    win_map = {1: 0.60, 2: 0.55, 3: 0.50, 4: 0.45, 5: 0.40}
    year_map = {1: 2010, 2: 2005, 3: 2000, 4: 1995, 5: 1990}

    filters = {
        "lob_in": ["COMMERCIAL PROPERTY"],
        "new_business_only": new_only,
        "loss_ratio_max": float(min(max_lr, lr_map[aggr])),
        "tiv_max": float(tiv_max),
        "premium_range": [float(prem_lo), float(prem_hi)],
        "min_winnability": float(win_map[aggr]),
        "min_year": int(year_map[aggr]),
        "good_construction_only": bool(strict_construction),
    }

    # objective weights
    if objective == "expected_premium":
        core = dict(w_prem=0.30, w_win=0.25, w_year=0.15, w_con=0.15, w_tiv=0.10, w_fresh=0.05)
    elif objective == "win_rate":
        core = dict(w_prem=0.20, w_win=0.35, w_year=0.15, w_con=0.15, w_tiv=0.10, w_fresh=0.05)
    elif objective == "freshness":
        core = dict(w_prem=0.15, w_win=0.25, w_year=0.10, w_con=0.10, w_tiv=0.05, w_fresh=0.35)
    else:  # balance
        core = dict(w_prem=0.25, w_win=0.25, w_year=0.15, w_con=0.15, w_tiv=0.10, w_fresh=0.10)

    span = max(1.0, prem_hi - prem_lo)
    mid_lo = prem_lo + 0.30 * span
    mid_hi = prem_lo + 0.70 * span

    weights = {
        **core,
        "premium_lo": float(prem_lo),
        "premium_mid_lo": float(mid_lo),
        "premium_mid_hi": float(mid_hi),
        "premium_hi": float(prem_hi),
        "tiv_hi": float(tiv_max),
    }

    return {"filters": filters, "weights": _normalize_weights(weights)}


