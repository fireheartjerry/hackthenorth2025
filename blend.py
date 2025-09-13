def blend_modes(left: dict, right: dict, t: float) -> dict:
    """
    Linear blend of two mode dicts {filters, weights} by t in [0,1].
    Only numeric-compatible fields are blended; others prefer right when t>=0.5.
    """
    t = max(0.0, min(1.0, float(t)))
    out = {"filters": {}, "weights": {}}

    # Filters
    lf = (left or {}).get("filters", {})
    rf = (right or {}).get("filters", {})
    keys = set(lf.keys()) | set(rf.keys())
    for k in keys:
        lv = lf.get(k)
        rv = rf.get(k)
        if isinstance(lv, (int, float)) and isinstance(rv, (int, float)):
            out["filters"][k] = (1 - t) * float(lv) + t * float(rv)
        elif k == "premium_range" and isinstance(lv, (list, tuple)) and isinstance(rv, (list, tuple)) and len(lv) == 2 and len(rv) == 2:
            lo = (1 - t) * float(lv[0]) + t * float(rv[0])
            hi = (1 - t) * float(lv[1]) + t * float(rv[1])
            if hi < lo:
                lo, hi = hi, lo
            out["filters"][k] = [lo, hi]
        else:
            out["filters"][k] = rv if t >= 0.5 else lv

    # Weights
    lw = (left or {}).get("weights", {})
    rw = (right or {}).get("weights", {})
    keys = set(lw.keys()) | set(rw.keys())
    for k in keys:
        lv = lw.get(k)
        rv = rw.get(k)
        if isinstance(lv, (int, float)) and isinstance(rv, (int, float)):
            out["weights"][k] = (1 - t) * float(lv) + t * float(rv)
        else:
            out["weights"][k] = rv if t >= 0.5 else lv

    return out


