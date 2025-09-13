from flask import Flask, request, jsonify, render_template, make_response
import os, json, time, datetime as dt
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from collections import Counter

# Custom modules
from utils_data import load_df
from scoring import score_row
from guidelines import classify_for_mode_row
from modes import MODES
from gemini_test import summarize_dataframe, seeds_from_answers, propose_with_gemini
from utils_json import df_records_to_builtin, to_builtin
from percentile_modes import build_percentile_filters, summarize_percentiles

load_dotenv()

app = Flask(__name__)

# Custom Jinja2 filter for date formatting
@app.template_filter('dateformat')
def dateformat(value, format='%b %d, %Y'):
    """Format a date that could be a string or datetime object."""
    if not value:
        return '—'
    
    # If it's already a datetime object, use strftime directly
    if hasattr(value, 'strftime'):
        return value.strftime(format)
    
    # If it's a string, try to parse it first
    if isinstance(value, str):
        try:
            # Try parsing common datetime formats
            for fmt in ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                try:
                    dt_obj = dt.datetime.strptime(value, fmt)
                    return dt_obj.strftime(format)
                except ValueError:
                    continue
            # If parsing fails, return the string as-is
            return value
        except:
            return value
    
    return '—'

FEDERATO_TOKEN = os.environ.get("FEDERATO_TOKEN")
CLIENT_ID = os.environ.get("FEDERATO_CLIENT_ID")
CLIENT_SECRET = os.environ.get("FEDERATO_CLIENT_SECRET")
AUDIENCE = os.environ.get("FEDERATO_AUDIENCE", "https://product.federato.ai/core-api")
AUTH_URL = os.environ.get(
    "FEDERATO_AUTH_URL", "https://product-federato.us.auth0.com/oauth/token"
)
POLICIES_URL_TYPO = os.environ.get(
    "FEDERATO_POLICIES_URL_TYPO",
    "https://product.federato.ai/integrations-api/handlers/all-pollicies?outputOnly=true",
)
POLICIES_URL = os.environ.get(
    "FEDERATO_POLICIES_URL",
    "https://product.federato.ai/integrations-api/handlers/all-policies?outputOnly=true",
)
USE_LOCAL_DATA = os.environ.get("USE_LOCAL_DATA", "false").lower() == "true"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

CACHE_TTL = 300
_cache_data = {"ts": 0, "df": None, "raw": None}

ACCEPT_STATES = {"OH", "PA", "MD", "CO", "CA", "FL", "NC", "SC", "GA", "VA", "UT"}
TARGET_STATES = {"OH", "PA", "MD", "CO", "CA", "FL"}
ACCEPT_CONSTRUCTION = {
    "Joisted Masonry",
    "Non-Combustible",
    "Masonry Non-Combustible",
    "Fire Resistive",
    "Non Combustible",
    "Non Combustible/Steel",
}
CURRENT_YEAR = dt.datetime.now().year


def getAccessToken():
    if FEDERATO_TOKEN:
        return FEDERATO_TOKEN
    if not CLIENT_ID or not CLIENT_SECRET:
        return None
    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "audience": AUDIENCE,
        "grant_type": "client_credentials",
    }
    r = requests.post(
        AUTH_URL, json=payload, headers={"Content-Type": "application/json"}
    )
    r.raise_for_status()
    return r.json().get("access_token")


def fetchPolicies(access_token):
    h = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    r = requests.post(POLICIES_URL_TYPO, headers=h, json={})
    if r.status_code == 404 or r.status_code == 405:
        r = requests.post(POLICIES_URL, headers=h, json={})
    r.raise_for_status()
    return r.json()


def loadLocalData():
    if not os.path.exists("data.json"):
        return None
    with open("data.json", "r", encoding="utf-8") as f:
        return json.load(f)


def normalizePayload(raw):
    if isinstance(raw, dict) and "data" in raw:
        return raw["data"]
    if isinstance(raw, dict) and "output" in raw:
        o = raw["output"]
        if (
            isinstance(o, list)
            and len(o) > 0
            and isinstance(o[0], dict)
            and "data" in o[0]
        ):
            return o[0]["data"]
    return []


def toDataFrame(items):
    df = pd.DataFrame(items)
    if df.empty:
        return df
    # Ensure expected columns exist to avoid KeyErrors downstream
    expected_num = [
        "tiv",
        "total_premium",
        "winnability",
        "loss_value",
        "oldest_building",
    ]
    expected_str = [
        "account_name",
        "primary_risk_state",
        "line_of_business",
        "renewal_or_new_business",
        "construction_type",
    ]
    expected_id = ["id"]
    for c in expected_num:
        if c not in df.columns:
            df[c] = np.nan
    for c in expected_str:
        if c not in df.columns:
            df[c] = ""
    for c in expected_id:
        if c not in df.columns:
            df[c] = pd.NA
    if "loss_value" in df.columns:
        df["loss_value"] = pd.to_numeric(df["loss_value"], errors="coerce")
    for col in ["total_premium", "winnability", "tiv"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["created_at", "effective_date", "expiration_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "oldest_building" in df.columns:
        df["building_age"] = CURRENT_YEAR - pd.to_numeric(
            df["oldest_building"], errors="coerce"
        )
    else:
        df["building_age"] = np.nan
    return df


def get_role():
    """Return a simple role flag for insights tabs.
    Prefers explicit `?role=` query, then cookie, else defaults to 'uw'.
    """
    try:
        role_q = (request.args.get("role") or "").strip().lower()
        if role_q in ("uw", "leader", "admin"):
            return role_q
        cookie_val = (request.cookies.get("role") or "").strip().lower()
        if cookie_val in ("uw", "leader", "admin"):
            return cookie_val
    except Exception:
        pass
    return "uw"


def classifyRow(row):
    reasons = []
    status = "IN"
    score = 0

    lob = str(row.get("line_of_business", "") or "").strip().upper()
    if lob != "COMMERCIAL PROPERTY":
        reasons.append("Line of Business not acceptable")

    sub_type = str(row.get("renewal_or_new_business", "") or "").strip().upper()
    if sub_type == "RENEWAL":
        reasons.append("Renewal business not acceptable")

    st = str(row.get("primary_risk_state", "") or "").strip().upper()
    if st not in ACCEPT_STATES:
        reasons.append("Primary risk state not acceptable")

    tiv = row.get("tiv", np.nan)
    if pd.notna(tiv):
        if tiv > 150_000_000:
            reasons.append("TIV exceeds 150M cap")
        elif 50_000_000 <= tiv <= 100_000_000:
            score += 2
        elif tiv <= 150_000_000:
            score += 1

    premium = row.get("total_premium", np.nan)
    if pd.notna(premium):
        if premium < 50_000 or premium > 175_000:
            reasons.append("Premium outside acceptable range")
        elif 75_000 <= premium <= 100_000:
            score += 2
        else:
            score += 1

    # Building age/year checks per guidelines:
    # - Newer than 2010 => Target
    # - Newer than 1990 => Acceptable
    # - Older than 1990 => Not acceptable
    age = row.get("building_age", np.nan)
    year = row.get("oldest_building", np.nan)
    # Prefer year-based strict comparisons when available; otherwise derive from age
    if pd.notna(year):
        try:
            y = int(float(year))
            if y > 2010:
                score += 2
            elif y > 1990:
                score += 1
            else:
                reasons.append("Building age older than 1990")
        except Exception:
            # Fallback to age logic if parsing fails
            if pd.notna(age):
                if age < (CURRENT_YEAR - 2010):
                    score += 2
                elif age <= (CURRENT_YEAR - 1990):
                    score += 1
                else:
                    reasons.append("Building age older than 1990")
    else:
        if pd.notna(age):
            if age < (CURRENT_YEAR - 2010):
                score += 2
            elif age <= (CURRENT_YEAR - 1990):
                score += 1
            else:
                reasons.append("Building age older than 1990")

    ctype = str(row.get("construction_type", "") or "").strip()
    if ctype in ACCEPT_CONSTRUCTION:
        score += 1
    else:
        reasons.append("Construction type not acceptable")

    loss = row.get("loss_value", np.nan)
    if pd.notna(loss):
        # exact match fix: >= 100_000 is Not Acceptable
        if loss >= 100_000:
            reasons.append("Loss value >= 100k")
        else:
            score += 1

    if reasons:
        status = "OUT"
    else:
        status = "IN"
        if (
            st in TARGET_STATES
            and 75_000 <= (premium or 0) <= 100_000
            and 50_000_000 <= (tiv or 0) <= 100_000_000
            and (
                (pd.notna(year) and float(year) > 2010)
                or (pd.notna(age) and age < (CURRENT_YEAR - 2010))
            )
        ):
            status = "TARGET"

    w = row.get("winnability", np.nan)
    if pd.notna(w):
        if isinstance(w, (int, float)) and w > 1:
            w = w / 100.0
        w = max(0.0, min(1.0, float(w)))
    else:
        w = 0.5

    appetite_score = score
    priority_score = appetite_score * 0.6 + w * 0.4

    try:
        with open("scores_debug.txt", "a", encoding="utf-8") as f:
            f.write(
                f"id={row.get('id')}, status={status}, appetite_score={appetite_score}, priority_score={priority_score}, reasons={reasons}\n"
            )
    except Exception:
        pass

    return status, reasons, appetite_score, priority_score



def classifyDataFrame(df):
    if df.empty:
        return df
    statuses = []
    reasons = []
    appetites = []
    priorities = []
    for _, r in df.iterrows():
        s, rs, a, p = classifyRow(r)
        statuses.append(s)
        reasons.append(rs)
        appetites.append(a)
        priorities.append(p)
    df = df.copy()
    df["appetite_status"] = statuses
    df["appetite_reasons"] = reasons
    df["appetite_score"] = appetites
    df["priority_score"] = priorities
    return df


def refreshCache():
    now = time.time()
    if now - _cache_data["ts"] < CACHE_TTL and _cache_data["df"] is not None:
        return _cache_data["df"], _cache_data["raw"]
    raw = None
    if USE_LOCAL_DATA:
        raw = loadLocalData()
    else:
        token = getAccessToken()
        if token:
            try:
                raw = fetchPolicies(token)
            except Exception:
                raw = loadLocalData()
        else:
            raw = loadLocalData()
    items = normalizePayload(raw or {})
    df = toDataFrame(items)
    df = classifyDataFrame(df)
    _cache_data["ts"] = now
    _cache_data["df"] = df
    _cache_data["raw"] = raw
    return df, raw


def _safe_json_value(v):
    # Handle missing values first where applicable
    try:
        if pd.isna(v):
            return None
    except Exception:
        # pd.isna may return an array for list-like; ignore here
        pass

    # Datetime-like
    if isinstance(v, pd.Timestamp):
        return v.isoformat()

    # Numpy scalars
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        try:
            if np.isnan(v):
                return None
        except Exception:
            pass
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)

    # List-like / ndarray / set / tuple
    if isinstance(v, (list, tuple, set, np.ndarray)):
        return list(v)

    return v


def dfToDict(df):
    cols = [c for c in df.columns]
    records = []
    for _, r in df.iterrows():
        x = {c: _safe_json_value(r[c]) for c in cols}
        records.append(x)
    return records


@app.route("/")
def home():
    df, _ = refreshCache()
    total = len(df)
    in_ct = int((df["appetite_status"] == "IN").sum()) if not df.empty else 0
    tgt_ct = int((df["appetite_status"] == "TARGET").sum()) if not df.empty else 0
    out_ct = int((df["appetite_status"] == "OUT").sum()) if not df.empty else 0
    avg_premium = float(df["total_premium"].mean()) if not df.empty else 0.0
    return render_template(
        "index.html",
        total=total,
        in_ct=in_ct,
        tgt_ct=tgt_ct,
        out_ct=out_ct,
        avg_premium=avg_premium,
        full_width=True,
    )


@app.route("/submissions")
def submissions():
    df, _ = refreshCache()
    state = request.args.get("state")
    status = request.args.get("status")
    q = request.args.get("q")
    min_p = request.args.get("min_premium", type=float)
    max_p = request.args.get("max_premium", type=float)
    d = df
    if state:
        d = d[d["primary_risk_state"].astype(str).str.upper() == state.upper()]
    if status:
        d = d[d["appetite_status"] == status]
    if min_p is not None:
        d = d[d["total_premium"] >= min_p]
    if max_p is not None:
        d = d[d["total_premium"] <= max_p]
    if q:
        m = d["account_name"].astype(str).str.contains(q, case=False, na=False)
        d = d[m.fillna(False)]  # Handle NaN values explicitly
    # Sort: TARGET -> IN -> OUT, then by priority score desc, premium desc
    try:
        d = d.copy()
        d["__s"] = pd.Categorical(
            d["appetite_status"], categories=["TARGET", "IN", "OUT"], ordered=True
        )
        d = d.sort_values(["__s", "priority_score", "total_premium"], ascending=[True, False, False])
        d = d.drop(columns=["__s"])
    except Exception:
        d = d.sort_values(["priority_score"], ascending=False)
    rows = dfToDict(d)
    return render_template("submissions.html", rows=rows)


@app.route("/insights")
def insights():
    # Portfolio Insights landing with simple role-based access for tabs
    role = get_role()
    tabs = [
        {"id": "overview", "title": "Overview", "protected": False},
        {"id": "premium_by_state", "title": "Premium by State", "protected": False},
        {"id": "status_mix", "title": "Status Mix", "protected": False},
        {"id": "tiv_bands", "title": "TIV Bands", "protected": True},
        {"id": "geo", "title": "US Map", "protected": False},
        {"id": "scatter3d", "title": "3D Scatter", "protected": False},
        {"id": "flows", "title": "Flows (Sankey)", "protected": False},
    ]
    resp = make_response(render_template("insights.html", role=role, tabs=tabs))
    # Persist role if provided via query
    role_q = request.args.get("role")
    if role_q:
        resp.set_cookie("role", role, max_age=7*24*3600, httponly=False, samesite="Lax")
    return resp


@app.route("/detail/<int:pid>")
def detail(pid):
    mode = request.args.get("mode")
    if mode and mode in MODES:
        try:
            ddf = load_df("data.json")
            dd = ddf[ddf["id"] == pid]
            if dd.empty:
                return render_template("detail.html", item=None)
            scounts = Counter(ddf["primary_risk_state"].fillna("UNK"))
            preset = MODES[mode]
            r = dd.iloc[0].to_dict()
            status, reasons, s, pscore = classify_for_mode_row(
                r, preset["weights"], preset["filters"], scounts
            )
            r["appetite_status"] = status
            r["appetite_reasons"] = reasons
            r["priority_score"] = float(pscore)
            return render_template("detail.html", item=r)
        except Exception:
            pass
    df, _ = refreshCache()
    d = df[df["id"] == pid]
    if d.empty:
        return render_template("detail.html", item=None)
    item = dfToDict(d)[0]
    return render_template("detail.html", item=item)


@app.route("/api/policies")
def apiPolicies():
    df, raw = refreshCache()
    return jsonify({"count": len(df), "data": dfToDict(df)})


@app.route("/api/classified")
def apiClassified():
    df, _ = refreshCache()
    state = request.args.get("state")
    status = request.args.get("status")
    min_p = request.args.get("min_premium", type=float)
    max_p = request.args.get("max_premium", type=float)
    q = request.args.get("q")
    d = df
    if state:
        d = d[d["primary_risk_state"].astype(str).str.upper() == state.upper()]
    if status:
        d = d[d["appetite_status"] == status]
    if min_p is not None:
        d = d[d["total_premium"] >= min_p]
    if max_p is not None:
        d = d[d["total_premium"] <= max_p]
    if q:
        m = d["account_name"].astype(str).str.contains(q, case=False, na=False)
        d = d[m.fillna(False)]
    d = d.sort_values(["priority_score"], ascending=False)
    return jsonify({"count": len(d), "data": dfToDict(d)})


@app.route("/api/refresh", methods=["POST"]) 
def apiRefresh():
    # Reset cache and refetch so UI can manually refresh
    _cache_data["ts"] = 0
    df, _ = refreshCache()
    return jsonify({"ok": True, "count": len(df)})


def buildGuidelineSummary():
    t = []
    t.append("Line of Business must be Commercial Property.")
    t.append("Submission type must be New Business.")
    t.append(
        "Acceptable states: OH, PA, MD, CO, CA, FL, NC, SC, GA, VA, UT; targets include OH, PA, MD, CO, CA, FL."
    )
    t.append("TIV acceptable up to 150M; target 50M–100M; over 150M not acceptable.")
    t.append("Premium acceptable 50k–175k; target 75k–100k; outside not acceptable.")
    t.append(
        "Building age newer than 1990 acceptable; newer than 2010 target; older than 1990 not acceptable."
    )
    t.append(
        "Construction acceptable when >50% JM, Non-Combustible/Steel, Masonry Non-Combustible, Fire Resistive."
    )
    t.append("Loss value must be < 100k.")
    return " ".join(t)


def buildModeGuidelineSummary(mode_name: str):
    p = MODES.get(mode_name)
    if not p:
        return "Mode not found; using default heuristic filters and weights."
    f = p.get("filters", {})
    parts = [f"Mode {mode_name} — Dynamic guideline summary:"]
    if f.get("lob_in"):
        parts.append(f"Allowed LOB: {', '.join(f['lob_in'])}.")
    if f.get("new_business_only"):
        parts.append("Only NEW_BUSINESS submissions.")
    if "loss_ratio_max" in f:
        parts.append(f"Loss ratio ≤ {f['loss_ratio_max']:.2f}.")
    if "tiv_max" in f:
        parts.append(f"TIV ≤ ${f['tiv_max']:,}.")
    if "premium_range" in f:
        lo, hi = f["premium_range"]
        parts.append(f"Premium in ${lo:,}–${hi:,} targetable window.")
    if "min_winnability" in f:
        parts.append(f"Winnability ≥ {f['min_winnability']:.2f}.")
    if "min_year" in f:
        parts.append(f"Building year ≥ {int(f['min_year'])} (or missing).")
    if f.get("good_construction_only"):
        parts.append("Acceptable construction types only.")
    parts.append(
        "Weighted by premium, winnability, year, construction, TIV, freshness; premium sweet spot defined by mid band."
    )
    return " ".join(parts)


def makePolicyDict(pid):
    df, _ = refreshCache()
    d = df[df["id"] == pid]
    if d.empty:
        return None
    return dfToDict(d)[0]


@app.route("/api/explain/<int:pid>", methods=["POST"])
def apiExplain(pid):
    mode = request.args.get("mode")
    if mode and mode in MODES:
        ddf = load_df("data.json")
        d = ddf[ddf["id"] == pid]
        if d.empty:
            return jsonify({"ok": False, "error": "Not found"}), 404
        preset = MODES[mode]
        scounts = Counter(ddf["primary_risk_state"].fillna("UNK"))
        r = d.iloc[0].to_dict()
        status, reasons, s, pscore = classify_for_mode_row(
            r, preset["weights"], preset["filters"], scounts
        )
        r["appetite_status"] = status
        r["appetite_reasons"] = reasons
        r["priority_score"] = float(pscore)
        item = r
        rule_text = buildModeGuidelineSummary(mode)
    else:
        item = makePolicyDict(pid)
        if not item:
            return jsonify({"ok": False, "error": "Not found"}), 404
        rule_text = buildGuidelineSummary()
    raw_reason = item.get("appetite_reasons", [])
    label = item.get("appetite_status", "IN")
    if not OPENAI_API_KEY:
        s = f"{label}: " + (
            "; ".join(raw_reason)
            if raw_reason
            else "Meets appetite and target criteria"
        )
        return jsonify({"ok": True, "explanation": s})
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"You are an underwriting assistant. Explain in one concise sentence why this submission is {label}. Use the guidelines and the fields. Guidelines: {rule_text}. Fields: {json.dumps(item, default=str)}."
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=120,
        )
        txt = resp.choices[0].message.content.strip()
        return jsonify({"ok": True, "explanation": txt})
    except Exception as e:
        s = f"{label}: " + ("; ".join(raw_reason) if raw_reason else "Meets appetite and target criteria")
        return jsonify({"ok": True, "explanation": s, "note": "AI fallback"})


@app.route("/api/nlq", methods=["POST"])
def apiNlq():
    data = request.get_json(silent=True) or {}
    q = (data.get("q") or "").strip()
    if not q:
        return jsonify({"filters": {}, "note": "Empty query"})
    if not OPENAI_API_KEY:
        st = None
        status = None
        min_p = None
        max_p = None
        for tok in q.split():
            if tok.upper() in ACCEPT_STATES:
                st = tok.upper()
        if "out" in q.lower():
            status = "OUT"
        elif "target" in q.lower():
            status = "TARGET"
        elif "in" in q.lower():
            status = "IN"
        return jsonify(
            {
                "filters": {
                    "state": st,
                    "status": status,
                    "min_premium": min_p,
                    "max_premium": max_p,
                },
                "note": "Heuristic",
            }
        )
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    sys = "Extract filters from a natural-language underwriting query. Respond as JSON with keys: state (2-letter or null), status (IN|OUT|TARGET|null), min_premium (number|null), max_premium (number|null), search (string|null)."
    u = f"Query: {q}"
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": u}],
            temperature=0.1,
            max_tokens=120,
        )
        txt = resp.choices[0].message.content.strip()
        try:
            parsed = json.loads(txt)
        except Exception:
            parsed = {
                "state": None,
                "status": None,
                "min_premium": None,
                "max_premium": None,
                "search": None,
            }
        return jsonify({"filters": parsed})
    except Exception:
        # Heuristic fallback on failure
        st = None
        status = None
        min_p = None
        max_p = None
        for tok in q.split():
            if tok.upper() in ACCEPT_STATES:
                st = tok.upper()
        if "out" in q.lower():
            status = "OUT"
        elif "target" in q.lower():
            status = "TARGET"
        elif "in" in q.lower():
            status = "IN"
        return jsonify(
            {
                "filters": {
                    "state": st,
                    "status": status,
                    "min_premium": min_p,
                    "max_premium": max_p,
                    "search": None,
                },
                "note": "Heuristic fallback",
            }
        )


@app.route("/api/modes")
def api_modes():
    # Return an array of objects so the UI can render key/label pairs
    arr = [
        {"key": k, "label": k.replace("_", " ").title()} for k in MODES.keys()
    ]
    return jsonify(arr)


@app.route("/api/mode/percentiles")
def api_mode_percentiles():
    try:
        df = load_df("data.json")
        if df.empty:
            return jsonify({"ok": True, "summary": {}})
        s = summarize_percentiles(df)
        return jsonify({"ok": True, "summary": to_builtin(s)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


def apply_hard_filters(df, f):
    d = df.copy()
    if "lob_in" in f:
        d = d[d["line_of_business"].str.upper().isin([x.upper() for x in f["lob_in"]])]
    if f.get("new_business_only"):
        d = d[d["renewal_or_new_business"].str.upper()=="NEW_BUSINESS"]
    if "loss_ratio_max" in f:
        d = d[(d["loss_ratio"].notna()) & (d["loss_ratio"]<=f["loss_ratio_max"])]
    if "tiv_max" in f:
        d = d[d["tiv"]<=f["tiv_max"]]
    if "premium_range" in f:
        lo, hi = f["premium_range"]
        d = d[(d["total_premium"]>=lo) & (d["total_premium"]<=hi)]
    if "min_winnability" in f:
        d = d[d["winnability"]>=f["min_winnability"]]
    if "max_fresh_days" in f and f.get("max_fresh_days") is not None:
        # keep only fresh submissions where we have the metric
        d = d[(d["fresh_days"].notna()) & (d["fresh_days"] <= f["max_fresh_days"])]
    if "min_year" in f:
        d = d[(d["building_year"]>=f["min_year"]) | (d["building_year"].isna())]
    if f.get("good_construction_only"):
        d = d[d["is_good_construction"]==True]
    if "fresh_days_max" in f:
        d = d[d["fresh_days"]<=f["fresh_days_max"]]
    return d


def sanitize_row(row):
    out = {}
    for k, v in row.items():
        if isinstance(v, (np.integer, np.int64, np.int32)):
            out[k] = int(v)
        elif isinstance(v, (np.floating, np.float64, np.float32)):
            out[k] = float(v)
        elif isinstance(v, (pd.Timestamp, dt.datetime)):
            out[k] = None if pd.isna(v) else v.isoformat()
        elif v is None:
            out[k] = None
        elif isinstance(v, float) and np.isnan(v):
            out[k] = None
        else:
            out[k] = v
    return out


def generate_explanation(r, qs):
    parts = []
    prem = r.get("total_premium")
    if prem is not None and not (isinstance(prem, float) and np.isnan(prem)):
        if prem >= qs["premium"].loc[0.9]:
            parts.append(f"premium ${prem:,.0f} is in top 10%")
        elif prem >= qs["premium"].loc[0.5]:
            parts.append(f"premium ${prem:,.0f} above median")
    tiv = r.get("tiv")
    if tiv is not None and not (isinstance(tiv, float) and np.isnan(tiv)):
        if tiv <= 150_000_000:
            parts.append(f"TIV ${tiv:,.0f} ≤ cap")
    win = r.get("winnability")
    if win is not None and not (isinstance(win, float) and np.isnan(win)):
        med_win = qs["winnability"].loc[0.5]
        if win >= med_win:
            parts.append(f"winnability {win:.2f} ≥ median")
    if not parts:
        return "No clear reasons available."
    prefix = (
        "In Appetite because "
        if r.get("appetite_status") != "OUT"
        else "Out of Appetite because "
    )
    return prefix + ", ".join(parts) + "."


@app.route("/api/classified_mode")
def api_classified_mode():
    """Percentile-driven modes with legacy preset fallback (e.g., custom).

    Supports overrides via query params: top_pct, max_lr_pct, min_win_pct, max_tiv_pct, fresh_pct.
    Also supports UI filters: state, status, min_premium, max_premium, q.
    """
    mode = request.args.get("mode", "balanced_growth")
    df = load_df("data.json")
    if df.empty:
        return jsonify({"count": 0, "data": []})

    preset = MODES.get(mode, MODES["balanced_growth"])

    qs = {
        "premium": df["total_premium"].quantile([0.99, 0.9, 0.8, 0.5, 0.4, 0.2]),
        "winnability": df["winnability"].quantile([0.99, 0.8, 0.5, 0.4]),
        "loss_ratio": df["loss_ratio"].quantile([0.4, 0.7, 0.9]),
        "fresh_days": df["fresh_days"].quantile([0.5]),
    }
    mode_filters = {
        "unicorn_hunting": {
            "premium_range": [qs["premium"].loc[0.99], df["total_premium"].max()],
            "min_winnability": qs["winnability"].loc[0.99],
            "loss_ratio_max": qs["loss_ratio"].loc[0.4],
        },
        "balanced_growth": {
            "premium_range": [qs["premium"].loc[0.8], df["total_premium"].max()],
            "min_winnability": qs["winnability"].loc[0.8],
            "loss_ratio_max": qs["loss_ratio"].loc[0.7],
        },
        "loose_fits": {
            "premium_range": [qs["premium"].loc[0.2], df["total_premium"].max()],
            "min_winnability": qs["winnability"].loc[0.4],
            "loss_ratio_max": qs["loss_ratio"].loc[0.9],
        },
        "turnaround_bets": {
            "premium_range": [df["total_premium"].min(), qs["premium"].loc[0.4]],
            "min_winnability": qs["winnability"].loc[0.4],
            "loss_ratio_max": 1.0,
            "fresh_days_max": qs["fresh_days"].loc[0.5],
        },
    }
    filters = mode_filters.get(mode, mode_filters["balanced_growth"])
    d = apply_hard_filters(df, filters)

    mode_expl = {
        "unicorn_hunting": "You're in Unicorn Mode — top 1% premiums & winnability",
        "balanced_growth": "You're in Balanced Mode — top 20% premiums, LR ≤ 70th percentile",
        "loose_fits": "You're in Loose Mode — wide net up to 80th percentile",
        "turnaround_bets": "You're in Turnaround Mode — smaller premiums but fresh",
    }

    # Optional UI filters
    state = request.args.get("state")
    status_filter = request.args.get("status")
    min_p = request.args.get("min_premium", type=float)
    max_p = request.args.get("max_premium", type=float)
    q = request.args.get("q")

    if state:
        d = d[d["primary_risk_state"].astype(str).str.upper() == state.upper()]
    if min_p is not None:
        d = d[d["total_premium"] >= min_p]
    if max_p is not None:
        d = d[d["total_premium"] <= max_p]
    if q:
        m = d["account_name"].astype(str).str.contains(q, case=False, na=False)
        d = d[m.fillna(False)]

    # Score and rank using existing mode weights
    scounts = Counter(d["primary_risk_state"].fillna("UNK"))
    rows = []
    for r in d.to_dict(orient="records"):
        stat, reasons, s, pscore = classify_for_mode_row(
            r, preset["weights"], filters, scounts
        )
        r["appetite_status"] = stat
        r["appetite_reasons"] = reasons
        r["priority_score"] = float(pscore)
        r["mode_score"] = round(float(pscore), 4)
        r["appetite_explanation"] = generate_explanation(r, qs)
        rows.append(sanitize_row(r))

    # Filter by status if requested
    if status_filter:
        rows = [r for r in rows if str(r.get("appetite_status")) == status_filter]

    # Sorting — support explicit sort_by/sort_dir like preset path
    sort_by = request.args.get("sort_by")
    sort_dir = (request.args.get("sort_dir") or "desc").lower()
    reverse = sort_dir != "asc"

    def sort_key(row, key):
        if key == "appetite_status":
            if reverse:
                order_map = {"TARGET": 0, "IN": 1, "OUT": 2}
            else:
                order_map = {"OUT": 0, "IN": 1, "TARGET": 2}
            return order_map.get(str(row.get("appetite_status") or ""), 99)
        v = row.get(key)
        if isinstance(v, str):
            return v.lower()
        try:
            if v is None:
                return float("-inf") if reverse else float("inf")
            import math
            if isinstance(v, float) and math.isnan(v):
                return float("-inf") if reverse else float("inf")
        except Exception:
            pass
        return v

    valid_keys = {
        "id",
        "account_name",
        "primary_risk_state",
        "line_of_business",
        "total_premium",
        "tiv",
        "winnability",
        "appetite_status",
        "priority_score",
        "mode_score",
    }

    if sort_by in valid_keys:
        rows.sort(key=lambda r: sort_key(r, sort_by), reverse=reverse)
    else:
        rows.sort(key=lambda r: r.get("priority_score"), reverse=True)

    return jsonify({
        "count": len(rows),
        "data": rows,
        "mode": mode,
        "mode_explanation": mode_expl.get(mode, ""),
    })


@app.route("/api/persona/seed", methods=["POST"])
def api_persona_seed():
    try:
        data = request.get_json(silent=True) or {}
        answers = data.get("answers") or {}
        df = load_df("data.json")
        # Build seed deterministically from answers
        seed = seeds_from_answers(answers)
        # Register as custom mode
        MODES["custom"] = seed
        return jsonify({"ok": True, "mode": "custom", "seed": seed})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/persona/propose", methods=["POST"])
def api_persona_propose():
    try:
        data = request.get_json(silent=True) or {}
        answers = data.get("answers") or {}
        goal_text = data.get("goal_text") or ""
        # Load dataset and summarize
        df = load_df("data.json")
        summary = summarize_dataframe(df)
        # Seed and propose
        seed = seeds_from_answers(answers)
        proposal = propose_with_gemini(summary, seed, goal_text)
        # Register proposal as custom mode
        MODES["custom"] = proposal
        return jsonify({
            "ok": True,
            "mode": "custom",
            "seed": seed,
            "proposal": proposal,
            "debug_summary": summary,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


def _apply_common_filters(df):
    d = df
    state = request.args.get("state")
    status = request.args.get("status")
    lob = request.args.get("lob")
    min_p = request.args.get("min_premium", type=float)
    max_p = request.args.get("max_premium", type=float)
    q = request.args.get("q")
    if state:
        d = d[d["primary_risk_state"].astype(str).str.upper() == state.upper()]
    if status:
        d = d[d["appetite_status"].astype(str).str.upper() == status.upper()]
    if lob:
        d = d[d["line_of_business"].astype(str).str.upper() == lob.upper()]
    if min_p is not None:
        d = d[d["total_premium"] >= min_p]
    if max_p is not None:
        d = d[d["total_premium"] <= max_p]
    if q:
        m = d["account_name"].astype(str).str.contains(q, case=False, na=False)
        d = d[m.fillna(False)]
    return d


@app.route("/api/aggregate")
def api_aggregate():
    """Generic aggregation endpoint for Portfolio Insights.
    Params:
      - group: column to group by (default primary_risk_state)
      - metric: one of count | sum:total_premium | sum:tiv | avg:total_premium
      - series_by: optional column to pivot series (e.g., appetite_status)
      - common filters: state,status,lob,min_premium,max_premium,q
    """
    df, _ = refreshCache()
    if df.empty:
        return jsonify({"labels": [], "series": []})
    group = request.args.get("group", "primary_risk_state")
    metric = request.args.get("metric", "count")
    series_by = request.args.get("series_by")
    d = _apply_common_filters(df)

    # Validate columns
    cols = set(d.columns)
    if group not in cols:
        return jsonify({"error": f"Unknown group '{group}'"}), 400
    if series_by and series_by not in cols:
        return jsonify({"error": f"Unknown series_by '{series_by}'"}), 400

    # Build aggregation
    def agg_series(frame):
        if metric == "count":
            return frame.size5
        if metric.startswith("sum:"):
            col = metric.split(":", 1)[1]
            if col not in cols:
                return 0
            return frame[col].sum(min_count=1)
        if metric.startswith("avg:"):
            col = metric.split(":", 1)[1]
            if col not in cols:
                return 0
            return frame[col].mean()
        # default to count
        return frame.size

    if series_by:
        piv = (
            d.groupby([group, series_by])
            .apply(lambda x: agg_series(x))
            .unstack(fill_value=0)
        )
        piv = piv.sort_index()
        labels = [str(x) for x in list(piv.index)]
        series = []
        for col in piv.columns:
            series.append({
                "name": str(col),
                "data": [float(v) if v == v else 0.0 for v in list(piv[col].values)],
            })
    else:
        agg = d.groupby(group).apply(lambda x: agg_series(x)).sort_values(ascending=False)
        labels = [str(k) for k in list(agg.index)]
        series = [{
            "name": metric,
            "data": [float(v) if v == v else 0.0 for v in list(agg.values)],
        }]

    return jsonify({
        "labels": labels,
        "series": series,
        "meta": {"group": group, "metric": metric, "series_by": series_by},
    })


@app.route("/api/underlying")
def api_underlying():
    """Return underlying rows for a chart segment.
       Accepts same filters plus optional label (value for the group) and series_val.
    """
    df, _ = refreshCache()
    if df.empty:
        return jsonify({"count": 0, "data": []})
    group = request.args.get("group", "primary_risk_state")
    label = request.args.get("label")
    series_by = request.args.get("series_by")
    series_val = request.args.get("series_val")
    d = _apply_common_filters(df)
    if label is not None and group in d.columns:
        d = d[d[group].astype(str) == label]
    if series_by and series_val is not None and series_by in d.columns:
        d = d[d[series_by].astype(str) == series_val]
    d = d.sort_values(["priority_score"], ascending=False)
    return jsonify({"count": len(d), "data": dfToDict(d)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
