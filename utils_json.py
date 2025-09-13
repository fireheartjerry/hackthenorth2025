import datetime as _dt
import numpy as _np
import pandas as _pd


def _iso_or_none(x):
    if isinstance(x, _pd.Timestamp):
        # pandas NaT safely maps to None
        return None if _pd.isna(x) else x.to_pydatetime().isoformat()
    if isinstance(x, (_dt.datetime, _dt.date)):
        return x.isoformat()
    return x


def to_builtin(obj):
    """Recursively convert pandas/numpy/datetime objects into JSON-safe Python builtins.

    Handles: pandas.Timestamp/NaT, numpy scalars, datetime/date, NaN, lists/tuples/ndarrays, dicts.
    """
    if obj is None:
        return None
    # fast path for common primitives
    if isinstance(obj, (str, int, float, bool)):
        # Normalize NaN floats to None
        if isinstance(obj, float):
            try:
                if _np.isnan(obj):
                    return None
            except Exception:
                pass
        return obj

    # numpy scalar types
    if isinstance(obj, (_np.generic,)):
        try:
            return obj.item()
        except Exception:
            pass

    # pandas/py datetime
    if isinstance(obj, (_pd.Timestamp, _dt.datetime, _dt.date)):
        return _iso_or_none(obj)

    # sequences
    if isinstance(obj, (list, tuple)):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, (set,)):
        return [to_builtin(x) for x in sorted(list(obj))]
    if isinstance(obj, _np.ndarray):
        return [to_builtin(x) for x in obj.tolist()]

    # dict
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}

    # pandas NaT/NaN handling
    try:
        if obj is _pd.NaT or (isinstance(obj, float) and _np.isnan(obj)):
            return None
    except Exception:
        pass

    # safe fallback string representation
    return str(obj)


def df_records_to_builtin(df: _pd.DataFrame):
    """Like df.to_dict('records') but sanitized for Flask JSON serialization."""
    if df is None or getattr(df, "empty", True):
        return []
    recs = []
    for r in df.to_dict(orient="records"):
        recs.append(to_builtin(r))
    return recs

