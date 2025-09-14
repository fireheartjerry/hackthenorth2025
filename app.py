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
from ai_suggest import summarize_dataframe
from persona_seed import seeds_from_answers
from ai_constraints import propose_with_gemini
from gemini_test import explain_with_gemini, nlq_with_gemini
from utils_json import df_records_to_builtin, to_builtin
from percentile_modes import build_percentile_filters, summarize_percentiles
from blend import blend_modes
from chatbot import chatbot

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

# AI functionality now handled by Gemini
# GEMINI_API_KEY and GEMINI_MODEL are read directly in gemini_test.py

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
    """
    Enhanced appetite classification and priority scoring system.
    
    Returns (status, reasons, appetite_score, priority_score)
    
    Status Logic:
    - TARGET: Premium submissions in target states with optimal characteristics
    - IN: Acceptable submissions that meet core appetite requirements  
    - OUT: Submissions that fail critical requirements
    
    Priority Score Range: 0.0 - 10.0
    - Combines appetite alignment (40%), winnability (30%), premium size (20%), and freshness (10%)
    - Higher scores indicate higher priority for underwriter attention
    """
    reasons = []
    base_score = 5.0  # Start at middle score
    multipliers = []
    
    # === CRITICAL REQUIREMENTS (Automatic OUT if failed) ===
    critical_failures = []
    
    # Line of Business - Must be Commercial Property
    lob = str(row.get("line_of_business", "") or "").strip().upper()
    if lob != "COMMERCIAL PROPERTY":
        critical_failures.append("Line of Business not Commercial Property")
    
    # Submission Type - Prefer New Business
    sub_type = str(row.get("renewal_or_new_business", "") or "").strip().upper()
    if sub_type == "RENEWAL":
        critical_failures.append("Renewal business (prefer new business)")
        base_score -= 2.0  # Still allow but penalize
    elif sub_type == "NEW_BUSINESS":
        base_score += 1.0  # Bonus for new business
        reasons.append("New business (preferred)")
        
    # If critical failures, mark as OUT
    if critical_failures:
        return "OUT", critical_failures, 1.0, max(0.1, base_score * 0.2)
    
    # === STATE SCORING ===
    st = str(row.get("primary_risk_state", "") or "").strip().upper()
    if st in TARGET_STATES:
        base_score += 2.0
        multipliers.append(1.2)
        reasons.append(f"Target state: {st}")
    elif st in ACCEPT_STATES:
        base_score += 1.0
        reasons.append(f"Acceptable state: {st}")
    else:
        base_score -= 1.5
        reasons.append(f"Non-preferred state: {st}")
    
    # === TIV SCORING ===
    tiv = row.get("tiv", np.nan)
    if pd.notna(tiv):
        if 50_000_000 <= tiv <= 100_000_000:  # Target range
            base_score += 2.0
            multipliers.append(1.15)
            reasons.append("TIV in target range (50M-100M)")
        elif tiv <= 150_000_000:  # Acceptable range
            base_score += 1.0
            reasons.append("TIV within acceptable limits")
        elif tiv > 150_000_000:  # Over limit
            base_score -= 2.0
            reasons.append("TIV exceeds 150M limit")
    else:
        base_score -= 0.5
        reasons.append("TIV data missing")
    
    # === PREMIUM SCORING ===
    premium = row.get("total_premium", np.nan)
    premium_multiplier = 1.0
    if pd.notna(premium):
        if 75_000 <= premium <= 100_000:  # Target range
            base_score += 2.0
            premium_multiplier = 1.3
            reasons.append("Premium in target range (75K-100K)")
        elif 50_000 <= premium <= 175_000:  # Acceptable range
            base_score += 1.0
            premium_multiplier = 1.1
            reasons.append("Premium in acceptable range")
        elif premium < 50_000:
            base_score -= 1.0
            reasons.append("Premium below minimum (50K)")
        else:  # > 175K
            base_score -= 1.0
            reasons.append("Premium above preferred maximum")
        
        # Additional premium size bonus
        if premium >= 1_000_000:
            base_score += 1.5
            premium_multiplier *= 1.2
            reasons.append("High-value premium (1M+)")
        elif premium >= 500_000:
            base_score += 0.5
            reasons.append("Substantial premium (500K+)")
    else:
        base_score -= 1.0
        reasons.append("Premium data missing")
    
    # === BUILDING AGE SCORING ===
    age = row.get("building_age", np.nan)
    year = row.get("oldest_building", np.nan)
    
    if pd.notna(year):
        try:
            y = int(float(year))
            current_age = CURRENT_YEAR - y
            if y > 2010:  # Less than ~14 years old
                base_score += 2.0
                multipliers.append(1.1)
                reasons.append("Modern building (post-2010)")
            elif y > 1990:  # Less than ~34 years old
                base_score += 1.0
                reasons.append("Acceptable building age (post-1990)")
            else:  # Older than 34 years
                base_score -= 1.5
                reasons.append("Older building construction (pre-1990)")
        except Exception:
            base_score -= 0.5
            reasons.append("Building age data unclear")
    elif pd.notna(age):
        if age < 14:  # Equivalent to post-2010
            base_score += 2.0
            multipliers.append(1.1)
            reasons.append("Modern building (< 14 years)")
        elif age <= 34:  # Equivalent to post-1990
            base_score += 1.0
            reasons.append("Acceptable building age")
        else:
            base_score -= 1.5
            reasons.append("Older building (> 34 years)")
    else:
        base_score -= 0.5
        reasons.append("Building age unknown")
    
    # === CONSTRUCTION TYPE SCORING ===
    ctype = str(row.get("construction_type", "") or "").strip()
    if ctype in ACCEPT_CONSTRUCTION:
        base_score += 1.0
        reasons.append(f"Preferred construction: {ctype}")
    else:
        base_score -= 0.5
        reasons.append(f"Non-preferred construction type")
    
    # === LOSS HISTORY SCORING ===
    loss = row.get("loss_value", np.nan)
    if pd.notna(loss):
        if loss < 25_000:  # Very low losses
            base_score += 1.5
            multipliers.append(1.1)
            reasons.append("Excellent loss history (< 25K)")
        elif loss < 100_000:  # Acceptable losses
            base_score += 0.5
            reasons.append("Acceptable loss history (< 100K)")
        else:  # High losses
            base_score -= 2.0
            reasons.append("Concerning loss history (≥ 100K)")
    else:
        # No loss data - assume neutral
        reasons.append("Loss history unknown")
    
    # === WINNABILITY FACTOR ===
    w = row.get("winnability", np.nan)
    if pd.notna(w):
        if isinstance(w, (int, float)) and w > 1:
            w = w / 100.0  # Convert percentage to decimal
        w = max(0.0, min(1.0, float(w)))
        
        if w >= 0.8:  # 80%+ win probability
            base_score += 1.5
            multipliers.append(1.2)
            reasons.append("Very high win probability (80%+)")
        elif w >= 0.6:  # 60%+ win probability
            base_score += 1.0
            reasons.append("High win probability (60%+)")
        elif w >= 0.4:  # 40%+ win probability
            base_score += 0.5
            reasons.append("Moderate win probability")
        else:  # Low win probability
            base_score -= 0.5
            reasons.append("Lower win probability")
    else:
        w = 0.5  # Default assumption
        reasons.append("Win probability unknown (assumed 50%)")
    
    # === CALCULATE FINAL SCORES ===
    
    # Apply multipliers to base score
    appetite_score = base_score
    for mult in multipliers:
        appetite_score *= mult
    
    # Ensure appetite score is in reasonable range
    appetite_score = max(1.0, min(10.0, appetite_score))
    
    # Calculate composite priority score
    # 40% appetite alignment, 30% winnability, 20% premium size, 10% freshness
    freshness_factor = 1.0  # Could be enhanced with submission date analysis
    
    priority_score = (
        appetite_score * 0.4 +
        (w * 10.0) * 0.3 +  # Scale winnability to 0-10 range
        (min(premium / 200_000, 1.0) * 10.0 if pd.notna(premium) else 5.0) * 0.2 +  # Premium factor
        freshness_factor * 10.0 * 0.1
    )
    
    # Apply premium multiplier to final score
    priority_score *= premium_multiplier
    
    # Ensure priority score is in 0-10 range
    priority_score = max(0.1, min(10.0, priority_score))
    
    # === DETERMINE STATUS ===
    severe_issues = [r for r in reasons if any(keyword in r.lower() for keyword in 
                    ['exceeds', 'below minimum', 'above preferred maximum', 'concerning', 'non-preferred state'])]
    
    if len(severe_issues) >= 3 or appetite_score < 3.0:
        status = "OUT"
    elif (st in TARGET_STATES and 
          appetite_score >= 7.0 and 
          priority_score >= 7.0 and
          len(severe_issues) == 0):
        status = "TARGET"
    else:
        status = "IN"
    
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
    avg_premium_str = f"${avg_premium:,.0f}"
    # Percent breakdowns from current dataset
    def _pct(part, whole):
        try:
            return int(round((part / whole) * 100.0)) if whole else 0
        except Exception:
            return 0
    pct_tgt = _pct(tgt_ct, total)
    pct_in = _pct(in_ct, total)
    pct_out = _pct(out_ct, total)

    # 7-day vs prior 7-day deltas based on created_at
    trend_target = trend_in = trend_out = trend_total = 0
    if not df.empty and "created_at" in df.columns:
        now = pd.Timestamp.utcnow()
        w1_start = now - pd.Timedelta(days=7)
        w2_start = now - pd.Timedelta(days=14)
        recent = df[df["created_at"] >= w1_start]
        prior = df[(df["created_at"] >= w2_start) & (df["created_at"] < w1_start)]

        def _delta(cur, prev):
            try:
                if prev == 0:
                    return 0
                return int(round(((cur - prev) / prev) * 100.0))
            except Exception:
                return 0

        trend_total = _delta(len(recent), len(prior))
        trend_target = _delta(int((recent["appetite_status"] == "TARGET").sum()), int((prior["appetite_status"] == "TARGET").sum()))
        trend_in = _delta(int((recent["appetite_status"] == "IN").sum()), int((prior["appetite_status"] == "IN").sum()))
        trend_out = _delta(int((recent["appetite_status"] == "OUT").sum()), int((prior["appetite_status"] == "OUT").sum()))

    # Winnability average (normalized 0..1)
    avg_win = 0.0
    if not df.empty and "winnability" in df.columns:
        s = pd.to_numeric(df["winnability"], errors="coerce")
        try:
            s = s.apply(lambda x: np.nan if pd.isna(x) else (x/100.0 if x>1 else float(x)))
            avg_win = float(s.mean(skipna=True)) if len(s) else 0.0
        except Exception:
            avg_win = 0.0

    # Opportunities = targets + in-appetite
    opp_ct = int(tgt_ct + in_ct)
    return render_template(
        "index.html",
        total=total,
        in_ct=in_ct,
        tgt_ct=tgt_ct,
        out_ct=out_ct,
        avg_premium=avg_premium,
        avg_premium_str=avg_premium_str,
        pct_tgt=pct_tgt,
        pct_in=pct_in,
        pct_out=pct_out,
        trend_target=trend_target,
        trend_in=trend_in,
        trend_out=trend_out,
        trend_total=trend_total,
        avg_win=avg_win,
        opp_ct=opp_ct,
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
        {"id": "overview", "title": "Triage", "protected": False},
        {"id": "premium_by_state", "title": "Pricing", "protected": False},
        {"id": "status_mix", "title": "Exposure", "protected": False},
        {"id": "tiv_bands", "title": "Renewals", "protected": False},
        {"id": "mix", "title": "Mix", "protected": False},
        {"id": "risk_signals", "title": "Risk Signals", "protected": False},
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


@app.route("/api/priority-accounts")
def apiPriorityAccounts():
    """API endpoint for Priority Account Review - returns top priority submissions"""
    df, _ = refreshCache()
    
    # Get query parameters
    state = request.args.get("state")
    status = request.args.get("status")
    min_p = request.args.get("min_premium", type=float)
    max_p = request.args.get("max_premium", type=float)
    q = request.args.get("q")
    page = request.args.get("page", default=1, type=int)
    per_page = request.args.get("per_page", default=20, type=int)
    sort_by = request.args.get("sort_by", default="priority_score")
    sort_dir = request.args.get("sort_dir", default="desc")
    
    # Apply filters
    d = df.copy()
    
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
    
    # Focus on higher-priority submissions for priority review
    # Show submissions with priority_score >= 1.0 and preferably IN or TARGET status
    d = d[(d["priority_score"] >= 1.0) & (d["appetite_status"].isin(["TARGET", "IN", "OUT"]))]
    
    # Apply sorting
    ascending = sort_dir.lower() == "asc"
    if sort_by in d.columns:
        d = d.sort_values([sort_by], ascending=ascending)
    else:
        d = d.sort_values(["priority_score"], ascending=False)
    
    # Calculate pagination
    total_count = len(d)
    total_pages = (total_count + per_page - 1) // per_page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    # Get paginated data
    paginated_data = d.iloc[start_idx:end_idx]
    
    # Convert to response format
    response_data = []
    for _, row in paginated_data.iterrows():
        response_data.append({
            "id": int(row["id"]),
            "account_name": str(row["account_name"]),
            "appetite_status": str(row["appetite_status"]),
            "total_premium": float(row["total_premium"]),
            "primary_risk_state": str(row["primary_risk_state"]),
            "winnability": float(row["winnability"]) if pd.notna(row["winnability"]) else 0.0,
            "priority_score": float(row["priority_score"]) if pd.notna(row["priority_score"]) else 0.0,
            "tiv": float(row["tiv"]) if pd.notna(row["tiv"]) else 0.0,
            "line_of_business": str(row["line_of_business"]) if pd.notna(row["line_of_business"]) else "",
            "appetite_reasons": row.get("appetite_reasons", []) if isinstance(row.get("appetite_reasons"), list) else [],
            "created_at": str(row["created_at"]) if pd.notna(row["created_at"]) else "",
            "effective_date": str(row["effective_date"]) if pd.notna(row["effective_date"]) else "",
        })
    
    return jsonify({
        "data": response_data,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total_count": total_count,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        },
        "filters": {
            "state": state,
            "status": status,
            "min_premium": min_p,
            "max_premium": max_p,
            "search": q,
            "sort_by": sort_by,
            "sort_dir": sort_dir
        }
    })


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
    
    # Use Gemini for explanation
    result = explain_with_gemini(item, rule_text, label, raw_reason)
    
    response = {
        "ok": True, 
        "explanation": result["explanation"],
        "ai_used": result["ai_used"]
    }
    
    if result["fallback_reason"]:
        response["ai_note"] = result["fallback_reason"]
    
    return jsonify(response)


@app.route("/api/nlq", methods=["POST"])
def apiNlq():
    data = request.get_json(silent=True) or {}
    q = (data.get("q") or "").strip()
    if not q:
        return jsonify({"filters": {}, "ai_used": False, "ai_summary": "Empty query provided"})
    
    # Use Gemini for NLQ parsing
    result = nlq_with_gemini(q)
    
    response = {
        "filters": result["filters"],
        "ai_used": result["ai_used"],
        "ai_summary": result["ai_summary"]
    }
    
    if result["fallback_reason"]:
        response["ai_note"] = result["fallback_reason"]
    
    return jsonify(response)


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


@app.route("/api/mode/blend", methods=["POST"])
def api_mode_blend():
    try:
        data = request.get_json(silent=True) or {}
        left = str(data.get("left") or "balanced_growth")
        right = str(data.get("right") or "unicorn_hunting")
        t = float(data.get("t") or 0.5)
        lm = MODES.get(left)
        rm = MODES.get(right)
        if not lm or not rm:
            return jsonify({"ok": False, "error": "Unknown mode(s)"}), 400
        blended = blend_modes(lm, rm, t)
        MODES["blend"] = blended
        return jsonify({"ok": True, "mode": "blend", "blended": to_builtin(blended)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Chat with the AI assistant"""
    try:
        data = request.get_json(silent=True) or {}
        message = data.get("message", "").strip()
        session_id = data.get("session_id", "default")
        
        if not message:
            return jsonify({"ok": False, "error": "No message provided"}), 400
        
        result = chatbot.chat(message, session_id)
        return jsonify({"ok": True, **result})
        
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/chat/stream")
def api_chat_stream():
    """Server-sent events for streaming chat responses"""
    from flask import Response
    
    message = request.args.get("message", "").strip()
    session_id = request.args.get("session_id", "default")
    
    if not message:
        return Response("data: " + json.dumps({"type": "error", "content": "No message provided"}) + "\n\n", 
                       mimetype="text/plain")
    
    def generate():
        try:
            for chunk in chatbot.get_streaming_response(message, session_id):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/chat/execute", methods=["POST"])
def api_chat_execute():
    """Execute an action suggested by the chatbot"""
    try:
        data = request.get_json(silent=True) or {}
        action = data.get("action", {})
        
        if not action or "type" not in action:
            return jsonify({"ok": False, "error": "No valid action provided"}), 400
        
        action_type = action["type"]
        params = action.get("params", {})
        
        # Build URL parameters for the action
        url_params = []
        
        if action_type == "filter":
            for key, value in params.items():
                if value is not None:
                    url_params.append(f"{key}={value}")
        
        elif action_type == "mode":
            url_params.append(f"mode={params.get('mode', 'balanced')}")
        
        elif action_type == "search":
            url_params.append(f"q={params.get('q', '')}")
        
        elif action_type == "sort":
            url_params.append(f"sort_by={params.get('sort_by', 'priority_score')}")
            url_params.append(f"sort_dir={params.get('sort_dir', 'desc')}")
        
        # Return the URL parameters for the frontend to apply
        return jsonify({
            "ok": True,
            "action_type": action_type,
            "url_params": "&".join(url_params),
            "message": f"Applied {action_type} action successfully"
        })
        
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/chat/context", methods=["POST"])
def api_chat_context():
    """Update chatbot with current dashboard context"""
    try:
        data = request.get_json(silent=True) or {}
        
        # Update chatbot with current dashboard state
        context_info = {
            "current_mode": data.get("mode", "balanced"),
            "active_filters": data.get("filters", {}),
            "current_page": data.get("page", "dashboard"),
            "visible_submissions": data.get("submission_count", 0),
            "user_action": data.get("last_action", ""),
        }
        
        # Store context in chatbot (you could extend this)
        if hasattr(chatbot, 'current_context'):
            chatbot.current_context = context_info
        
        return jsonify({"ok": True, "message": "Context updated"})
        
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/chat/suggestions")
def api_chat_suggestions():
    """Get contextual suggestions based on current dashboard state"""
    try:
        df = load_df("data.json")
        if df.empty:
            return jsonify({"suggestions": []})
        
        suggestions = []
        
        # Analyze current data to provide smart suggestions
        if "total_premium" in df.columns:
            high_premium_count = len(df[df["total_premium"] > df["total_premium"].quantile(0.9)])
            if high_premium_count > 0:
                suggestions.append({
                    "text": f"Show me the {high_premium_count} highest premium submissions",
                    "action": "filter",
                    "icon": "💰"
                })
        
        if "primary_risk_state" in df.columns:
            top_state = df["primary_risk_state"].value_counts().index[0]
            state_count = df["primary_risk_state"].value_counts().iloc[0]
            suggestions.append({
                "text": f"Focus on {state_count} submissions in {top_state}",
                "action": "filter",
                "icon": "📍"
            })
        
        if "fresh_days" in df.columns:
            fresh_count = len(df[df["fresh_days"] <= 7])
            if fresh_count > 0:
                suggestions.append({
                    "text": f"Show me {fresh_count} submissions from this week",
                    "action": "filter",
                    "icon": "🆕"
                })
        
        # Mode suggestions
        suggestions.extend([
            {"text": "Switch to unicorn mode for premium opportunities", "action": "mode", "icon": "🦄"},
            {"text": "Explain current filtering criteria", "action": "explain", "icon": "💡"},
            {"text": "What are the trends in my portfolio?", "action": "analyze", "icon": "📈"}
        ])
        
        return jsonify({"suggestions": suggestions[:6]})  # Limit to 6 suggestions
        
    except Exception as e:
        return jsonify({"suggestions": [], "error": str(e)})


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
    """Percentile-driven modes and persona custom.

    Query params:
      - mode: one of legacy [unicorn_hunting, balanced_growth, loose_fits, turnaround_bets] or new [unicorn, balanced, loose, turnaround, custom]
      - overrides: top_pct, max_lr_pct, min_win_pct, max_tiv_pct, fresh_pct
      - filters: state, status, min_premium, max_premium, q
      - sorting: sort_by, sort_dir
    """
    mode = (request.args.get("mode") or "balanced_growth").strip()
    df = load_df("data.json")
    if df.empty:
        return jsonify({"count": 0, "data": []})

    # Map legacy/new names → percentile kind
    name_map = {
        "unicorn_hunting": "unicorn",
        "balanced_growth": "balanced",
        "loose_fits": "loose",
        "turnaround_bets": "turnaround",
        "unicorn": "unicorn",
        "balanced": "balanced",
        "loose": "loose",
        "turnaround": "turnaround",
    }

    if mode == "custom" and "custom" in MODES:
        # Persona-driven
        preset = MODES["custom"]
        filters = preset.get("filters", {})
        d = apply_hard_filters(df, filters)
        weights = preset.get("weights", MODES.get("balanced_growth", {}).get("weights", {}))
        mode_explanation = "Custom mode (persona-refined)"
    else:
        kind = name_map.get(mode, "balanced")
        # Collect overrides
        overrides = {}
        for k in ("top_pct", "max_lr_pct", "min_win_pct", "max_tiv_pct", "fresh_pct"):
            if k in request.args:
                overrides[k] = request.args.get(k)
        filters, _summary = build_percentile_filters(df, kind=kind, overrides=overrides)
        d = apply_hard_filters(df, filters)
        weights = MODES.get("balanced_growth", {}).get("weights", {})
        mode_explanation = f"{kind.title()} mode — dynamic percentile-based filters"

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

    # Score and rank
    scounts = Counter(d["primary_risk_state"].fillna("UNK"))
    rows = []
    for r in d.to_dict(orient="records"):
        stat, reasons, s, pscore = classify_for_mode_row(
            r, weights, filters, scounts
        )
        r["appetite_status"] = stat
        r["appetite_reasons"] = reasons
        r["priority_score"] = float(pscore)
        r["mode_score"] = round(float(pscore), 4)
        rows.append(sanitize_row(r))

    # Cut to top N% if provided in filters
    top_pct = None
    try:
        if isinstance(filters.get("top_pct"), (int, float)):
            top_pct = float(filters.get("top_pct"))
    except Exception:
        top_pct = None
    if top_pct:
        n = max(1, int(len(rows) * (top_pct / 100.0)))
        rows.sort(key=lambda r: r.get("priority_score") or 0.0, reverse=True)
        rows = rows[:n]

    # Filter by status if requested
    if status_filter:
        rows = [r for r in rows if str(r.get("appetite_status")) == status_filter]

    # Sorting
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
        "id", "account_name", "primary_risk_state", "line_of_business",
        "total_premium", "tiv", "winnability", "appetite_status",
        "priority_score", "mode_score",
    }
    if sort_by in valid_keys:
        rows.sort(key=lambda r: sort_key(r, sort_by), reverse=reverse)
    else:
        rows.sort(key=lambda r: r.get("priority_score"), reverse=True)

    return jsonify({
        "count": len(rows),
        "data": rows,
        "mode": mode,
        "mode_explanation": mode_explanation,
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
            return frame.size
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
