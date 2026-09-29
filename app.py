import hashlib
from html import escape
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.api_client import (
    predict_providers,
    get_investigations,
    create_investigation,
    update_investigation,
    get_prediction_runs,
    get_predictions_by_run,
)
from src.preprocessing import convert_date_columns, create_features
from src.model_input import provider_aggregation
from src.explainability.shap_explainer import FraudExplainer

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="FraudLens AI | Healthcare Risk Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# SESSION STATE
# ============================================================
DEFAULT_STATE = {
    "prediction_run_id": None,
    "aggregated": None,
    "source": None,
    "run_model": {},
    "risk_summary": {},
    "database_result": {},
    "upload_signature": None,
    "show_upload": False,
    "autoload_tried": False,
    "flash": None,
}
for _k, _v in DEFAULT_STATE.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ============================================================
# MODEL
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_pipeline.pkl"

try:
    pipeline = joblib.load(MODEL_PATH)
    features = pipeline["features"]
    model = pipeline["model"]
    model_version = pipeline.get("model_version", "unknown")
    threshold = float(pipeline.get("threshold", 0.45))
    model_type = pipeline.get("model_type", "RandomForestClassifier")
except Exception as exc:
    st.error(f"Unable to load model artifact: {exc}")
    st.stop()


@st.cache_resource
def get_explainer(_model, feature_names):
    return FraudExplainer(model=_model, features=list(feature_names))


# ============================================================
# HELPERS
# ============================================================
RISK_COLORS = {"High": "#fb3b64", "Medium": "#f5a524", "Low": "#12d6a0"}


def html(s: str):
    """Render HTML; collapsing whitespace stops Markdown treating blank lines as code."""
    st.markdown(" ".join(s.split()), unsafe_allow_html=True)


def flash(kind: str, message: str):
    """Message that survives st.rerun() (toasts are lost/hidden)."""
    st.session_state.flash = (kind, message)


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def risk_emoji(risk):
    return {"High": "🔴", "Medium": "🟠", "Low": "🟢"}.get(str(risk), "⚪")


def build_prediction_dataframe(predictions):
    cols = ["Provider", "Prediction", "Fraud Probability", "Risk Level", "Model Version",
            "Threshold", "Prediction ID", "Run ID", "Created At"]
    if not predictions:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame([
        {
            "Provider": str(i.get("provider_id", "")),
            "Prediction": safe_int(i.get("prediction")),
            "Fraud Probability": safe_float(i.get("fraud_probability")),
            "Risk Level": str(i.get("risk_level", "Unknown")),
            "Model Version": str(i.get("model_version", model_version)),
            "Threshold": safe_float(i.get("threshold", threshold), threshold),
            "Prediction ID": i.get("id"),
            "Run ID": i.get("run_id"),
            "Created At": i.get("created_at"),
        }
        for i in predictions
    ])


def render_table(df, max_rows=None, percent_columns=None):
    """Dark glass HTML table with risk badges and inline probability bars."""
    if df is None or df.empty:
        st.info("No records to display.")
        return
    t = df.head(max_rows).copy() if max_rows else df.copy()
    pct = set(percent_columns or [])
    head = "".join(f"<th>{escape(str(c))}</th>" for c in t.columns)
    body = []
    for _, row in t.iterrows():
        cells = []
        for col in t.columns:
            v = row[col]
            if pd.isna(v):
                cells.append("<td>—</td>")
            elif col in ("Risk Level", "risk_level"):
                cells.append(f'<td><span class="badge b-{escape(str(v).lower())}">{escape(str(v))}</span></td>')
            elif col in pct:
                p = min(max(safe_float(v), 0), 1)
                cells.append(f'<td><div class="pbar"><span>{p:.1%}</span><i style="width:{p*100:.1f}%"></i></div></td>')
            else:
                cells.append(f"<td>{escape(str(v))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown(
        f'<div class="table-wrap"><table class="glass-table"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def style_fig(fig, height=420):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c7cde0", family="Inter, Arial, sans-serif"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.07)", zerolinecolor="rgba(255,255,255,.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.07)", zerolinecolor="rgba(255,255,255,.12)")
    return fig


def latest_completed_run(runs):
    return next((r for r in runs if str(r.get("status", "")).lower() == "completed"), None)


def load_database_run(run=None):
    """Load an existing completed run from PostgreSQL. Never executes the ML pipeline."""
    try:
        if run is None:
            run = latest_completed_run(get_prediction_runs().get("runs", []))
        if not run:
            return False, "No completed prediction runs found."

        run_id = int(run["id"])
        predictions = get_predictions_by_run(run_id).get("predictions", [])
        if not predictions:
            return False, f"Run {run_id} contains no predictions."

        df = build_prediction_dataframe(predictions)
        st.session_state.prediction_run_id = run_id
        st.session_state.aggregated = df
        st.session_state.source = "PostgreSQL"
        st.session_state.risk_summary = {
            "high": safe_int(run.get("high_risk_count"), int((df["Risk Level"] == "High").sum())),
            "medium": safe_int(run.get("medium_risk_count"), int((df["Risk Level"] == "Medium").sum())),
            "low": safe_int(run.get("low_risk_count"), int((df["Risk Level"] == "Low").sum())),
        }
        st.session_state.run_model = {
            "version": run.get("model_version", model_version),
            "type": model_type,
            "threshold": safe_float(df["Threshold"].iloc[0], threshold),
        }
        st.session_state.database_result = {
            "success": True, "saved_count": len(df), "run_id": run_id, "source": "PostgreSQL",
        }
        return True, f"Run {run_id} loaded."
    except Exception as exc:
        return False, str(exc)


def file_signature(*files):
    h = hashlib.sha256()
    for f in files:
        if f is None:
            continue
        h.update(f.name.encode("utf-8"))
        h.update(f.getvalue())
    return h.hexdigest()


def run_new_prediction(beneficiary_file, inpatient_file, outpatient_file):
    """Preprocess -> features -> FastAPI -> PostgreSQL."""
    with st.status("Running healthcare fraud analysis...", expanded=True) as status:
        status.write("📥 Reading datasets...")
        beneficiary = pd.read_csv(beneficiary_file)
        inpatient = pd.read_csv(inpatient_file)
        outpatient = pd.read_csv(outpatient_file)

        status.write("📅 Converting date columns...")
        beneficiary = convert_date_columns(beneficiary, ["DOB", "DOD"])
        inpatient = convert_date_columns(
            inpatient, ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt"]
        )
        outpatient = convert_date_columns(outpatient, ["ClaimStartDt", "ClaimEndDt"])

        status.write("🔗 Combining claims and joining beneficiaries...")
        claims = pd.concat([inpatient, outpatient], ignore_index=True)
        claims = claims.merge(beneficiary, on="BeneID", how="left")

        status.write("🧬 Engineering provider-level features...")
        claims = create_features(claims)

        status.write("📊 Aggregating provider behavior...")
        aggregated = provider_aggregation(claims).fillna(0).reset_index(drop=True)
        X = aggregated[features]

        status.write(f"🤖 Sending {len(X):,} providers to FastAPI...")
        providers = aggregated["Provider"].astype(str).tolist()
        payload = [
            {"provider_id": p,
             "features": {k: (v.item() if hasattr(v, "item") else v) for k, v in rec.items()}}
            for p, rec in zip(providers, X.to_dict("records"))
        ]

        api_response = predict_providers(payload)
        predictions = api_response["predictions"]
        new_run_id = api_response.get("run_id")

        if new_run_id is None:
            raise ValueError("API did not return a run_id.")
        if len(predictions) != len(aggregated):
            raise ValueError("Prediction count does not match provider count.")

        persisted = get_predictions_by_run(int(new_run_id)).get("predictions", [])
        by_provider = {str(i.get("provider_id")): i for i in persisted}

        aggregated = aggregated.copy()
        aggregated["Prediction"] = [i["prediction"] for i in predictions]
        aggregated["Fraud Probability"] = [i["fraud_probability"] for i in predictions]
        aggregated["Risk Level"] = [i["risk_level"] for i in predictions]
        aggregated["Model Version"] = [i["model_version"] for i in predictions]
        aggregated["Threshold"] = [i["threshold"] for i in predictions]
        aggregated["Prediction ID"] = [by_provider.get(p, {}).get("id") for p in providers]
        aggregated["Created At"] = [by_provider.get(p, {}).get("created_at") for p in providers]
        aggregated["Run ID"] = new_run_id

        st.session_state.prediction_run_id = new_run_id
        st.session_state.aggregated = aggregated
        st.session_state.source = "New Analysis"
        st.session_state.risk_summary = api_response.get("risk_summary", {})
        st.session_state.run_model = api_response.get("model", {})
        st.session_state.database_result = api_response.get("database", {})
        st.session_state.upload_signature = file_signature(
            beneficiary_file, inpatient_file, outpatient_file
        )
        status.update(label=f"Analysis complete — Run {new_run_id}", state="complete", expanded=False)


def refresh_database_run():
    ok, message = load_database_run()
    flash("success" if ok else "warning",
          "Latest prediction run loaded from PostgreSQL." if ok else message)


def reset_current_analysis():
    for k, v in DEFAULT_STATE.items():
        st.session_state[k] = v
    st.session_state.autoload_tried = True  # don't silently reload right after a manual reset


# ============================================================
# STYLES  (dark glass command-center)
# ============================================================
st.markdown(r'''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
:root{
  --bg:#060810;--card:rgba(255,255,255,.045);--card-2:rgba(255,255,255,.07);
  --line:rgba(255,255,255,.09);--ink:#eef1fb;--muted:#8e97b3;
  --cyan:#22d3ee;--violet:#8b5cf6;--pink:#fb3b64;--amber:#f5a524;--green:#12d6a0;
}
/* Base — Material icon glyphs are deliberately left alone */
html,body,.stApp{font-family:Inter,Arial,sans-serif}
.stApp{
  color:var(--ink);color-scheme:dark;
  background:
    radial-gradient(900px 500px at 8% -8%,rgba(139,92,246,.30),transparent 60%),
    radial-gradient(800px 500px at 96% 2%,rgba(34,211,238,.18),transparent 60%),
    radial-gradient(700px 500px at 60% 110%,rgba(251,59,100,.14),transparent 60%),
    var(--bg)!important;
}
:where(.stApp) :where(p,span,li,label,h1,h2,h3,h4,h5,h6,div){color:var(--ink)}
:where(.stApp) :where([data-testid="stIconMaterial"]){font-family:'Material Symbols Rounded'!important}
[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],#MainMenu,footer{display:none!important}
.main .block-container{max-width:1560px!important;margin:0 auto!important;padding:22px 44px 60px!important}
h1,h2,h3,h4{letter-spacing:-.03em;font-weight:800}
h3{font-size:1.15rem!important;margin-top:.4rem!important}
hr{border-color:var(--line)!important}

/* Top bar + hero */
.topbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}
.brand{display:flex;align-items:center;gap:12px}
.brand-logo{width:40px;height:40px;border-radius:12px;display:grid;place-items:center;font-size:20px;
  background:linear-gradient(135deg,var(--violet),var(--cyan));box-shadow:0 0 28px rgba(139,92,246,.55)}
.brand-name{font-size:20px;font-weight:900;letter-spacing:-.05em}
.brand-sub{font-size:12px;color:var(--muted)}
.live{display:flex;align-items:center;gap:8px;padding:8px 14px;border-radius:999px;background:var(--card);
  border:1px solid var(--line);font-size:12px;font-weight:700;backdrop-filter:blur(12px)}
.live i{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 0 0 rgba(18,214,160,.7);animation:ping 1.8s infinite}
@keyframes ping{70%{box-shadow:0 0 0 9px rgba(18,214,160,0)}100%{box-shadow:0 0 0 0 rgba(18,214,160,0)}}
.hero{position:relative;overflow:hidden;padding:34px 36px;border-radius:26px;margin-bottom:18px;
  background:linear-gradient(135deg,rgba(139,92,246,.22),rgba(34,211,238,.08) 60%,rgba(255,255,255,.03));
  border:1px solid rgba(255,255,255,.12);box-shadow:0 20px 60px rgba(0,0,0,.45)}
.hero:before{content:"";position:absolute;width:420px;height:420px;right:-120px;top:-220px;border-radius:50%;
  background:conic-gradient(from 90deg,var(--violet),var(--cyan),var(--pink),var(--violet));filter:blur(70px);opacity:.35;animation:spin 14s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.hero>*{position:relative;z-index:1}
.kicker{font-size:11px;font-weight:800;letter-spacing:.2em;text-transform:uppercase;color:var(--cyan)!important}
.hero-title{font-size:44px;line-height:1.05;font-weight:900;letter-spacing:-.06em;margin-top:10px;
  background:linear-gradient(90deg,#fff,#c4b5fd 45%,#67e8f9);-webkit-background-clip:text;background-clip:text;
  -webkit-text-fill-color:transparent;color:transparent!important}
.hero-text{max-width:760px;margin-top:12px;font-size:14px;line-height:1.65;color:#b6bdd6!important}
.pills{display:flex;gap:8px;flex-wrap:wrap;margin-top:18px}
.pills span{padding:7px 12px;border-radius:999px;font-size:10px;font-weight:800;letter-spacing:.08em;
  background:rgba(255,255,255,.07);border:1px solid var(--line);color:#dbe1f5!important}

/* Toolbar container */
.st-key-toolbar{background:var(--card)!important;border:1px solid var(--line)!important;border-radius:18px!important;
  padding:10px 14px!important;margin-bottom:16px;backdrop-filter:blur(14px)}

/* Metric cards */
.metric-card{position:relative;min-height:122px;padding:18px 20px;border-radius:20px;background:var(--card);
  border:1px solid var(--line);backdrop-filter:blur(14px);overflow:hidden;transition:transform .2s,border-color .2s}
.metric-card:hover{transform:translateY(-3px);border-color:rgba(255,255,255,.2)}
.metric-card:after{content:"";position:absolute;left:0;right:0;top:0;height:2px;background:var(--accent,var(--cyan))}
.metric-card.hot{background:linear-gradient(135deg,rgba(251,59,100,.28),rgba(139,92,246,.16));
  border-color:rgba(251,59,100,.45);box-shadow:0 0 40px rgba(251,59,100,.22)}
.metric-label{font-size:12px;font-weight:700;color:var(--muted)!important;text-transform:uppercase;letter-spacing:.08em}
.metric-value{font-size:34px;font-weight:900;letter-spacing:-.05em;margin-top:8px}
.metric-delta{font-size:12px;color:var(--muted)!important;margin-top:4px}

/* Cards */
.section-card{padding:20px;border-radius:20px;background:var(--card);border:1px solid var(--line);
  backdrop-filter:blur(14px);margin-bottom:14px}
.section-card h3{margin:0 0 6px}
.alert-card{padding:14px 18px;border-radius:16px;font-size:13px;line-height:1.55;
  background:linear-gradient(90deg,rgba(251,59,100,.18),rgba(251,59,100,.05));border:1px solid rgba(251,59,100,.4)}
.small-muted{font-size:12px;color:var(--muted)!important}
.pulse-label{font-size:12px;text-transform:uppercase;letter-spacing:.1em;font-weight:800;color:var(--muted)!important}
.pulse-value{font-size:36px;font-weight:900;letter-spacing:-.05em;margin-top:4px}
.pulse-track{height:8px;background:rgba(255,255,255,.08);border-radius:99px;margin-top:14px;overflow:hidden}
.pulse-fill{height:100%;border-radius:99px;box-shadow:0 0 14px currentColor}

/* Table */
.table-wrap{overflow:auto;border-radius:18px;border:1px solid var(--line);background:var(--card);backdrop-filter:blur(14px)}
.glass-table{width:100%;border-collapse:collapse;font-size:13px}
.glass-table th{position:sticky;top:0;text-align:left;padding:12px 14px;font-size:11px;font-weight:800;letter-spacing:.09em;
  text-transform:uppercase;color:var(--muted)!important;background:rgba(255,255,255,.05);border-bottom:1px solid var(--line);white-space:nowrap}
.glass-table td{padding:12px 14px;border-bottom:1px solid rgba(255,255,255,.05);white-space:nowrap;color:var(--ink)}
.glass-table tr:last-child td{border-bottom:none}
.glass-table tbody tr:hover td{background:rgba(255,255,255,.04)}
.badge{display:inline-block;padding:4px 11px;border-radius:999px;font-size:11px;font-weight:800;letter-spacing:.04em}
.b-high{background:rgba(251,59,100,.16);color:#ff7f98!important;border:1px solid rgba(251,59,100,.45)}
.b-medium{background:rgba(245,165,36,.14);color:#ffc55e!important;border:1px solid rgba(245,165,36,.4)}
.b-low{background:rgba(18,214,160,.13);color:#4fe8c0!important;border:1px solid rgba(18,214,160,.4)}
.b-unknown{background:rgba(255,255,255,.08);color:var(--muted)!important;border:1px solid var(--line)}
.pbar{position:relative;min-width:120px;height:22px;border-radius:8px;background:rgba(255,255,255,.06);overflow:hidden}
.pbar i{position:absolute;left:0;top:0;bottom:0;background:linear-gradient(90deg,var(--violet),var(--pink));opacity:.55}
.pbar span{position:relative;z-index:1;display:block;padding:3px 9px;font-size:12px;font-weight:700}

/* Native widgets */
.stButton>button,.stDownloadButton>button{min-height:42px!important;border-radius:12px!important;font-weight:700!important;font-size:13px!important;
  background:var(--card-2)!important;border:1px solid var(--line)!important;color:var(--ink)!important;transition:all .18s!important}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--cyan)!important;box-shadow:0 0 22px rgba(34,211,238,.28)!important;transform:translateY(-1px)}
.stButton>button[kind="primary"],.stDownloadButton>button{background:linear-gradient(135deg,var(--violet),#3b82f6 60%,var(--cyan))!important;border:none!important;color:#fff!important;box-shadow:0 8px 24px rgba(99,102,241,.4)!important}
.stButton>button *,.stDownloadButton>button *{color:inherit!important}
.stButton>button:disabled{opacity:.45!important}
[data-testid="stWidgetLabel"] p{font-size:12px!important;font-weight:700!important;color:var(--muted)!important;text-transform:uppercase;letter-spacing:.07em}
.stTextInput input,.stTextArea textarea,[data-baseweb="select"]>div,[data-baseweb="input"]{background:rgba(255,255,255,.06)!important;
  color:var(--ink)!important;border-color:var(--line)!important;border-radius:12px!important}
.stTextInput input::placeholder{color:#6d7594!important}
[data-baseweb="popover"],[data-baseweb="menu"]{background:#11142a!important;border:1px solid var(--line)!important;border-radius:12px!important}
[data-baseweb="menu"] *{color:var(--ink)!important;background:transparent!important}
[data-baseweb="menu"] li:hover{background:rgba(139,92,246,.25)!important}
[data-baseweb="tag"]{background:rgba(139,92,246,.3)!important}
.stFileUploader section{background:rgba(255,255,255,.04)!important;border:1px dashed rgba(255,255,255,.22)!important;border-radius:16px!important}
[data-testid="stExpander"]{background:var(--card)!important;border:1px solid var(--line)!important;border-radius:18px!important;margin-bottom:14px}
[data-testid="stExpander"] summary p{font-size:14px!important;font-weight:800!important}
.stTabs [data-baseweb="tab-list"]{gap:6px;border:none!important;flex-wrap:wrap}
.stTabs [data-baseweb="tab"]{height:42px;padding:0 18px;border-radius:12px;font-size:13px;font-weight:700;background:var(--card)!important;border:1px solid var(--line)}
.stTabs [data-baseweb="tab"] p{color:var(--muted)!important}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,rgba(139,92,246,.55),rgba(34,211,238,.35))!important;border-color:rgba(255,255,255,.25)!important}
.stTabs [aria-selected="true"] p{color:#fff!important}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none!important}
[data-testid="stMetric"]{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:14px 16px;backdrop-filter:blur(14px)}
[data-testid="stMetricLabel"] p{color:var(--muted)!important;font-size:12px!important}
[data-testid="stMetricValue"]{font-weight:800!important}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:16px;overflow:hidden}
[data-testid="stAlert"]{border-radius:14px;background:var(--card)!important;border:1px solid var(--line)}
.stCaption,.stCaption *{color:var(--muted)!important}
.footer{margin-top:40px;padding:30px 0 6px;border-top:1px solid var(--line);text-align:center}
.footer-brand{font-size:16px;font-weight:900}
.footer div{color:var(--muted)!important;font-size:12px;margin-top:6px}
@media(max-width:900px){.main .block-container{padding:14px 14px 40px!important}.hero-title{font-size:30px}.live{display:none}.hero{padding:24px}}
</style>
''', unsafe_allow_html=True)

# ============================================================
# HEADER + HERO
# ============================================================
html('''
<div class="topbar">
  <div class="brand"><div class="brand-logo">🛡️</div>
    <div><div class="brand-name">FraudLens AI</div><div class="brand-sub">Healthcare risk intelligence</div></div></div>
  <div class="live"><i></i>All systems operational</div>
</div>
<div class="hero">
  <div class="kicker">Command center</div>
  <div class="hero-title">See the signal<br>before the case.</div>
  <div class="hero-text">Provider-level machine learning signals, persisted prediction runs, explainability
  and investigation workflows — in one focused workspace.</div>
  <div class="pills"><span>● MODEL ONLINE</span><span>FASTAPI CONNECTED</span><span>POSTGRESQL PERSISTENCE</span></div>
</div>
''')

if st.session_state.flash:
    _kind, _msg = st.session_state.flash
    st.session_state.flash = None
    getattr(st, _kind, st.info)(_msg)

# ============================================================
# AUTO-LOAD LATEST RUN (once per session)
# ============================================================
if st.session_state.aggregated is None and not st.session_state.autoload_tried:
    st.session_state.autoload_tried = True
    with st.spinner("Connecting to the latest prediction run..."):
        load_database_run()

# ============================================================
# NEW ANALYSIS
# ============================================================
with st.expander("＋  Run a New Healthcare Analysis", expanded=st.session_state.show_upload):
    html('''<div class="section-card"><b>Upload the three source datasets to create a new prediction run.</b>
    <div class="small-muted">Runs preprocessing → feature engineering → FastAPI → Random Forest → PostgreSQL.</div></div>''')

    beneficiary_file = st.file_uploader("Beneficiary Dataset", type=["csv"], key="beneficiary_upload")
    inpatient_file = st.file_uploader("Inpatient Claims Dataset", type=["csv"], key="inpatient_upload")
    outpatient_file = st.file_uploader("Outpatient Claims Dataset", type=["csv"], key="outpatient_upload")

    if all([beneficiary_file, inpatient_file, outpatient_file]):
        already_processed = st.session_state.upload_signature == file_signature(
            beneficiary_file, inpatient_file, outpatient_file
        )
        if already_processed:
            st.success("These files have already been processed in this session.")
        if st.button("⚡ Analyze Healthcare Claims", type="primary",
                     use_container_width=True, disabled=already_processed):
            try:
                run_new_prediction(beneficiary_file, inpatient_file, outpatient_file)
                st.session_state.show_upload = False
                flash("success", "New prediction run completed and saved to PostgreSQL.")
                st.rerun()
            except Exception as exc:
                st.error(f"Prediction pipeline failed: {exc}")

# ============================================================
# CURRENT DATA
# ============================================================
aggregated = st.session_state.aggregated
prediction_run_id = st.session_state.prediction_run_id
source = st.session_state.source
run_model = st.session_state.run_model

if aggregated is None or aggregated.empty:
    st.warning("No prediction data is available yet. Upload datasets and run an analysis.")
    st.stop()

aggregated = aggregated.copy()
for _col, _default in (("Provider", ""), ("Fraud Probability", 0.0), ("Risk Level", "Unknown")):
    if _col not in aggregated.columns:
        aggregated[_col] = _default
aggregated["Fraud Probability"] = pd.to_numeric(aggregated["Fraud Probability"], errors="coerce").fillna(0)
aggregated["Risk Level"] = aggregated["Risk Level"].fillna("Unknown")

top_providers = aggregated.sort_values("Fraud Probability", ascending=False).copy()
high_risk = int((aggregated["Risk Level"] == "High").sum())
medium_risk = int((aggregated["Risk Level"] == "Medium").sum())
low_risk = int((aggregated["Risk Level"] == "Low").sum())
provider_count = len(aggregated)
average_probability = float(aggregated["Fraud Probability"].mean())
max_probability = float(aggregated["Fraud Probability"].max())
high_risk_rate = high_risk / provider_count if provider_count else 0
current_threshold = safe_float(run_model.get("threshold", threshold), threshold)
current_model_version = run_model.get("version", model_version)

# ============================================================
# TOOLBAR + METRICS
# ============================================================
with st.container(border=True, key="toolbar"):
    tc = st.columns([1.2, 2.6, 1, 1.2])
    scope_mode = tc[0].selectbox("Scope", ["All providers", "High risk", "Above threshold"], key="scope_mode")
    quick_search = tc[1].text_input("Search", placeholder="Search provider ID…", key="quick_search")
    tc[2].markdown('<div style="height:29px"></div>', unsafe_allow_html=True)
    if tc[2].button("↻ Refresh", use_container_width=True, key="top_refresh"):
        refresh_database_run()
        st.rerun()
    tc[3].markdown('<div style="height:29px"></div>', unsafe_allow_html=True)
    if tc[3].button("＋ New analysis", use_container_width=True, key="top_new"):
        st.session_state.show_upload = True
        st.rerun()

filtered_top = top_providers.copy()
if quick_search:
    filtered_top = filtered_top[
        filtered_top["Provider"].astype(str).str.contains(quick_search, case=False, na=False, regex=False)
    ]
if scope_mode == "High risk":
    filtered_top = filtered_top[filtered_top["Risk Level"] == "High"]
elif scope_mode == "Above threshold":
    filtered_top = filtered_top[filtered_top["Fraud Probability"] >= current_threshold]

mc = st.columns(4)
cards = [
    ("hot", "High-risk providers", f"{high_risk:,}", f"{high_risk_rate:.1%} of current run", "#fb3b64"),
    ("", "Providers analyzed", f"{provider_count:,}", f"Run #{prediction_run_id}", "#22d3ee"),
    ("", "Average probability", f"{average_probability:.1%}", f"Model threshold {current_threshold:.2f}", "#8b5cf6"),
    ("", "Peak probability", f"{max_probability:.1%}", "Highest provider signal", "#f5a524"),
]
for col, (cls, label, value, delta, accent) in zip(mc, cards):
    col.markdown(
        f'<div class="metric-card {cls}" style="--accent:{accent}"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-delta">{delta}</div></div>',
        unsafe_allow_html=True,
    )
st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab_command, tab_investigation, tab_explain, tab_analytics, tab_history, tab_data = st.tabs(
    ["Command Center", "Investigations", "Explainability", "Analytics", "Run History", "Data Explorer"]
)

# ---------------- COMMAND CENTER ----------------
with tab_command:
    if high_risk > 0:
        html(f'''<div class="alert-card"><b>🚨 Risk signal detected</b> — {high_risk:,} providers are classified
        as High Risk. Open the Investigations tab to prioritize operational review.</div>''')
    st.write("")

    left, right = st.columns([1.5, 1])
    with left:
        st.markdown("### Highest-risk providers")
        cols = [c for c in ["Provider", "Fraud Probability", "Risk Level", "Prediction ID"] if c in filtered_top.columns]
        render_table(filtered_top[cols].head(12), max_rows=12, percent_columns=["Fraud Probability"])
    with right:
        st.markdown("### Risk distribution")
        fig = go.Figure(go.Pie(
            labels=["High", "Medium", "Low"], values=[high_risk, medium_risk, low_risk], hole=0.68,
            textinfo="label+percent", marker=dict(colors=[RISK_COLORS[k] for k in ("High", "Medium", "Low")],
                                                   line=dict(color="#060810", width=3)),
        ))
        style_fig(fig, 400).update_layout(legend=dict(orientation="h", y=-0.05), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Risk pulse")
    for col, (label, count, color) in zip(st.columns(3), [
        ("High risk", high_risk, RISK_COLORS["High"]),
        ("Medium risk", medium_risk, RISK_COLORS["Medium"]),
        ("Low risk", low_risk, RISK_COLORS["Low"]),
    ]):
        ratio = count / provider_count if provider_count else 0
        col.markdown(
            f'<div class="section-card"><div class="pulse-label">{label}</div>'
            f'<div class="pulse-value">{count:,}</div>'
            f'<div class="pulse-track"><div class="pulse-fill" style="width:{ratio*100:.2f}%;background:{color};color:{color}"></div></div>'
            f'<div class="small-muted" style="margin-top:9px">{ratio:.2%} of analyzed providers</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Provider risk profile")
    profile_options = top_providers["Provider"].astype(str).head(100).tolist()
    selected_profile = st.selectbox("Inspect a provider", profile_options, key="command_provider_profile")
    prow = top_providers[top_providers["Provider"].astype(str) == selected_profile].iloc[0]
    pprob = safe_float(prow.get("Fraud Probability"))
    prisk = str(prow.get("Risk Level", "Unknown"))
    pc = st.columns([1.3, 1, 1, 1])
    with pc[0]:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=pprob * 100, number={"suffix": "%", "font": {"size": 34, "color": "#eef1fb"}},
            gauge={"axis": {"range": [0, 100], "tickcolor": "#5b6382"},
                   "bar": {"color": RISK_COLORS.get(prisk, "#8b5cf6"), "thickness": .3},
                   "bgcolor": "rgba(255,255,255,.04)", "borderwidth": 0,
                   "steps": [{"range": [0, 30], "color": "rgba(18,214,160,.14)"},
                             {"range": [30, 70], "color": "rgba(245,165,36,.14)"},
                             {"range": [70, 100], "color": "rgba(251,59,100,.16)"}]},
        ))
        style_fig(gauge, 200).update_layout(margin=dict(l=20, r=20, t=20, b=5))
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
    pc[1].metric("Risk Level", f"{risk_emoji(prisk)} {prisk}")
    pc[2].metric("Prediction ID", str(prow.get("Prediction ID", "—")))
    pc[3].metric("Decision", "Above threshold" if pprob >= current_threshold else "Below threshold")

    st.markdown("### Model snapshot")
    sc = st.columns(4)
    sc[0].metric("Model", model_type)
    sc[1].metric("Version", str(current_model_version))
    sc[2].metric("Threshold", f"{current_threshold:.2f}")
    sc[3].metric("Peak Probability", f"{max_probability:.2%}")
    st.caption(f"Data source: {source or 'Unknown'} • Run #{prediction_run_id}")

# ---------------- INVESTIGATIONS ----------------
with tab_investigation:
    st.markdown("### Investigation operations center")
    try:
        resp = get_investigations()
        inv = resp.get("investigations", []) if isinstance(resp, dict) else resp
        investigation_df = pd.DataFrame(inv)
    except Exception as exc:
        st.error(f"Unable to load investigations: {exc}")
        investigation_df = pd.DataFrame()

    for _c in ("status", "priority"):
        if _c not in investigation_df.columns:
            investigation_df[_c] = None

    cc = st.columns(5)
    cc[0].metric("Total Cases", len(investigation_df))
    cc[1].metric("Open", int((investigation_df["status"] == "Open").sum()))
    cc[2].metric("Under Review", int((investigation_df["status"] == "Under Review").sum()))
    cc[3].metric("Resolved", int((investigation_df["status"] == "Resolved").sum()))
    cc[4].metric("Priority", int(investigation_df["priority"].isin(["High", "Critical"]).sum()))
    st.markdown("---")

    create_col, manage_col = st.columns([1, 1.25])

    with create_col:
        st.markdown("### Create investigation")
        provider_labels = {}
        for _, r in top_providers[["Provider", "Fraud Probability", "Risk Level"]].head(500).iterrows():
            provider_labels[f"{risk_emoji(r['Risk Level'])} {r['Provider']} — {safe_float(r['Fraud Probability']):.2%}"] = str(r["Provider"])

        if provider_labels:
            selected_provider = provider_labels[
                st.selectbox("Provider", list(provider_labels.keys()), key="investigation_provider")
            ]
            create_status = st.selectbox("Initial Status", ["Open", "Under Review"], key="investigation_status")
            create_priority = st.selectbox("Priority", ["Low", "Normal", "High", "Critical"], index=2,
                                           key="investigation_priority")
            assigned_to = st.text_input("Assigned Investigator", value="Fraud Investigation Team",
                                        key="investigation_assignee")
            notes = st.text_area("Investigation Notes",
                                 value=f"Review provider patterns from prediction Run {prediction_run_id}.",
                                 height=120, key="investigation_notes")
            if st.button("🚨 Create Case", type="primary", use_container_width=True, key="create_case"):
                try:
                    result = create_investigation(
                        provider_id=selected_provider,
                        run_id=int(prediction_run_id),
                        status=create_status,
                        priority=create_priority,
                        assigned_to=assigned_to.strip() or None,
                        notes=notes.strip() or None,
                    )
                    flash("success", f"Investigation #{result.get('investigation_id')} created for provider {selected_provider}.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to create investigation: {exc}")

    with manage_col:
        st.markdown("### Manage investigation")
        if investigation_df.empty:
            st.info("No investigation cases have been created yet.")
        else:
            options = {
                f"#{r['id']} • {r['provider_id']} • Run {r.get('run_id', 'N/A')} • {r['status']}": r["id"]
                for _, r in investigation_df.iterrows()
            }
            case_id = options[st.selectbox("Select Case", list(options.keys()), key="manage_case")]
            case = investigation_df[investigation_df["id"] == case_id].iloc[0]

            ic = st.columns(4)
            ic[0].metric("Provider", str(case.get("provider_id", "N/A")))
            ic[1].metric("Run", str(case.get("run_id", "N/A")))
            ic[2].metric("Risk", str(case.get("risk_level", "N/A")))
            ic[3].metric("Probability", f"{safe_float(case.get('fraud_probability')):.2%}")

            status_options = ["Open", "Under Review", "Resolved"]
            priority_options = ["Low", "Normal", "High", "Critical"]
            cur_status = str(case.get("status", "Open"))
            cur_priority = str(case.get("priority", "Normal"))
            if cur_status not in status_options:
                cur_status = "Open"
            if cur_priority not in priority_options:
                cur_priority = "Normal"
            cur_assigned, cur_notes = case.get("assigned_to", ""), case.get("notes", "")

            new_status = st.selectbox("Status", status_options, index=status_options.index(cur_status),
                                      key=f"status_{case_id}")
            new_priority = st.selectbox("Priority", priority_options, index=priority_options.index(cur_priority),
                                        key=f"priority_{case_id}")
            new_assigned = st.text_input("Assigned To", value="" if pd.isna(cur_assigned) else str(cur_assigned),
                                         key=f"assigned_{case_id}")
            new_notes = st.text_area("Notes", value="" if pd.isna(cur_notes) else str(cur_notes),
                                     height=120, key=f"notes_{case_id}")

            if st.button("💾 Save Investigation", type="primary", use_container_width=True,
                         key=f"save_case_{case_id}"):
                try:
                    update_investigation(
                        investigation_id=int(case_id),
                        status=new_status,
                        priority=new_priority,
                        assigned_to=new_assigned.strip() or None,
                        notes=new_notes.strip() or None,
                    )
                    flash("success", f"Investigation #{case_id} updated.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to update investigation: {exc}")

    st.markdown("### Investigation queue")
    if investigation_df.empty:
        st.info("Investigation queue is empty.")
    else:
        queue_cols = [c for c in ["id", "provider_id", "prediction_id", "run_id", "fraud_probability",
                                  "risk_level", "model_version", "status", "priority", "assigned_to",
                                  "created_at"] if c in investigation_df.columns]
        render_table(investigation_df[queue_cols], max_rows=50, percent_columns=["fraud_probability"])

# ---------------- EXPLAINABILITY ----------------
with tab_explain:
    st.markdown("### Explainable AI laboratory")
    has_features = all(f in aggregated.columns for f in features)

    if not has_features:
        html('''<div class="section-card"><h3>🔐 Feature-aware mode required</h3>
        <div class="small-muted">SHAP needs the provider-level feature matrix. A run restored from PostgreSQL
        only stores prediction results, and feature values are never invented. Upload the source datasets
        and run a new analysis to unlock explanations.</div></div>''')
    else:
        choices = top_providers["Provider"].astype(str).head(50).tolist()
        sel = st.selectbox("Select provider", choices, key="shap_provider")
        rows = aggregated[aggregated["Provider"].astype(str) == sel]

        if not rows.empty:
            sel_features = rows[features].iloc[0]
            sel_prob = safe_float(rows["Fraud Probability"].iloc[0])
            sel_risk = str(rows["Risk Level"].iloc[0])

            html(f'''<div class="section-card"><h3>Provider {escape(sel)}</h3>
            <span class="badge b-{escape(sel_risk.lower())}">{escape(sel_risk)}</span>
            &nbsp; Model probability <b>{sel_prob:.2%}</b> &nbsp;·&nbsp; Run <b>{prediction_run_id}</b></div>''')

            try:
                explainer = get_explainer(model, tuple(features))
                with st.spinner("Calculating SHAP explanation..."):
                    explanation = explainer.explain(sel_features)

                if explanation is not None:
                    top_exp = explanation.head(12).copy()
                    render_table(top_exp[["Feature", "Feature Value", "SHAP Value", "Impact"]], max_rows=12)
                    fig = px.bar(top_exp.sort_values("SHAP Value"), x="SHAP Value", y="Feature",
                                 orientation="h", title="Feature contributions to model output",
                                 hover_data=["Feature Value", "Impact"],
                                 color="SHAP Value", color_continuous_scale=["#12d6a0", "#8b5cf6", "#fb3b64"])
                    fig.add_vline(x=0, line_width=1, line_color="rgba(255,255,255,.4)")
                    style_fig(fig, 520).update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(f"SHAP explanation error: {exc}")

# ---------------- ANALYTICS ----------------
with tab_analytics:
    st.markdown("### Risk intelligence analytics")
    a1, a2 = st.columns(2)

    with a1:
        fig = px.histogram(aggregated, x="Fraud Probability", nbins=30, title="Provider probability distribution",
                           color_discrete_sequence=["#8b5cf6"])
        fig.add_vline(x=current_threshold, line_dash="dash", line_color="#22d3ee",
                      annotation_text="Decision threshold", annotation_font_color="#22d3ee")
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    with a2:
        top20 = top_providers.head(20).copy()
        top20["Provider"] = top20["Provider"].astype(str)
        fig = px.bar(top20, x="Fraud Probability", y="Provider", color="Risk Level", orientation="h",
                     title="Top 20 provider risk signals", color_discrete_map=RISK_COLORS,
                     hover_data=["Prediction ID"] if "Prediction ID" in top20.columns else None)
        fig.update_yaxes(type="category", autorange="reversed")
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    st.markdown("### Risk mix")
    mix = pd.DataFrame({
        "Risk Level": ["High", "Medium", "Low"],
        "Share": [(n / provider_count if provider_count else 0) for n in (high_risk, medium_risk, low_risk)],
    })
    fig = px.bar(mix, x="Share", y="Risk Level", orientation="h", color="Risk Level",
                 color_discrete_map=RISK_COLORS, text=mix["Share"].map("{:.1%}".format))
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style_fig(fig, 260).update_layout(showlegend=False), use_container_width=True)

    st.markdown("### Operational indicators")
    oc = st.columns(4)
    oc[0].metric("Providers Above Threshold", f"{int((aggregated['Fraud Probability'] >= current_threshold).sum()):,}")
    oc[1].metric("High Risk Rate", f"{high_risk_rate:.2%}")
    oc[2].metric("Mean Probability", f"{average_probability:.2%}")
    oc[3].metric("Peak Probability", f"{max_probability:.2%}")

# ---------------- RUN HISTORY ----------------
with tab_history:
    st.markdown("### Prediction run history")
    try:
        runs = get_prediction_runs().get("runs", [])
        if not runs:
            st.info("No prediction runs found.")
        else:
            run_df = pd.DataFrame(runs)
            latest = latest_completed_run(runs)
            if latest:
                html(f'''<div class="section-card"><h3>⚡ Latest completed run #{latest.get("id")}</h3>
                <div class="small-muted">{safe_int(latest.get("provider_count")):,} providers ·
                {safe_int(latest.get("high_risk_count")):,} high ·
                {safe_int(latest.get("medium_risk_count")):,} medium ·
                {safe_int(latest.get("low_risk_count")):,} low ·
                Model {escape(str(latest.get("model_version", "N/A")))}</div></div>''')

            hist_cols = [c for c in ["id", "model_version", "provider_count", "high_risk_count",
                                     "medium_risk_count", "low_risk_count", "status", "started_at",
                                     "completed_at"] if c in run_df.columns]
            render_table(run_df[hist_cols], max_rows=50)

            st.markdown("### Load a historical run")
            completed = [r for r in runs if str(r.get("status", "")).lower() == "completed"]
            if completed:
                labels = {
                    f"Run {r['id']} — {safe_int(r.get('provider_count')):,} providers — {r.get('model_version', 'N/A')}": r
                    for r in completed
                }
                chosen = labels[st.selectbox("Select run", list(labels.keys()), key="historical_run")]
                if st.button("📥 Load Selected Run", type="primary", use_container_width=True, key="load_history_run"):
                    ok, message = load_database_run(chosen)
                    if ok:
                        flash("success", message)
                        st.rerun()
                    else:
                        st.error(message)
    except Exception as exc:
        st.error(f"Unable to load run history: {exc}")

# ---------------- DATA EXPLORER ----------------
with tab_data:
    st.markdown("### Provider data explorer")
    ec = st.columns([2, 2, 1.5])
    search = ec[0].text_input("Search provider", placeholder="Type a provider ID...", key="provider_search")
    risk_filter = ec[1].multiselect("Risk filter", ["High", "Medium", "Low"],
                                    default=["High", "Medium", "Low"], key="risk_filter")
    min_probability = ec[2].slider("Minimum probability", 0.0, 1.0, 0.0, 0.01, key="min_prob")

    explorer_df = aggregated.copy()
    if search:
        explorer_df = explorer_df[
            explorer_df["Provider"].astype(str).str.contains(search, case=False, na=False, regex=False)
        ]
    if risk_filter:
        explorer_df = explorer_df[explorer_df["Risk Level"].isin(risk_filter)]
    explorer_df = explorer_df[explorer_df["Fraud Probability"] >= min_probability]

    st.caption(f"Showing {len(explorer_df):,} of {len(aggregated):,} providers")

    display_df = explorer_df.copy()
    display_df["Fraud Probability"] = display_df["Fraud Probability"] * 100
    st.dataframe(
        display_df, use_container_width=True, hide_index=True, height=520,
        column_config={"Fraud Probability": st.column_config.ProgressColumn(
            "Fraud Probability", min_value=0, max_value=100, format="%.1f%%")},
    )
    st.download_button(
        "⬇️ Export Filtered Providers",
        data=explorer_df.to_csv(index=False),
        file_name=f"fraud_run_{prediction_run_id}_providers.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ============================================================
# FOOTER
# ============================================================
html('''<div class="footer"><div class="footer-brand">🛡️ FraudLens AI</div>
<div>Healthcare Fraud Analytics Platform</div>
<div>Machine learning risk intelligence · Explainability · Investigation workflow · PostgreSQL persistence</div>
<div>Developed by Appaji Gouda</div></div>''')