# app/streamlit_app.py
# ─────────────────────────────────────────────────────────────────────────────
# Professional Streamlit dashboard for loan eligibility prediction.
#
# Calls the FastAPI backend at localhost:8000.
# The frontend knows nothing about ML — it only sends HTTP requests.
# This separation means you can replace the backend model without
# changing any frontend code. That's clean architecture.
#
# Run with: streamlit run app/streamlit_app.py
# Requires: FastAPI backend running on port 8000
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time

# ── Page config — must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title = "Loan Eligibility Predictor",
    page_icon  = "🏦",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

API_URL   = "http://localhost:8000"
THRESHOLD = 0.77

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Card styling */
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px 0;
    }

    /* Decision banner */
    .approved-banner {
        background: linear-gradient(135deg, #00c896, #00a876);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin: 15px 0;
    }
    .rejected-banner {
        background: linear-gradient(135deg, #ff4d6d, #cc3055);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin: 15px 0;
    }

    /* Section headers */
    .section-header {
        font-size: 18px;
        font-weight: bold;
        color: #e0e0e0;
        border-bottom: 2px solid #2a2d3a;
        padding-bottom: 8px;
        margin: 20px 0 15px 0;
    }

    /* Factor pills */
    .factor-positive {
        background: rgba(0, 200, 150, 0.15);
        border: 1px solid #00c896;
        color: #00c896;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 13px;
        display: inline-block;
        margin: 3px;
    }
    .factor-negative {
        background: rgba(255, 77, 109, 0.15);
        border: 1px solid #ff4d6d;
        color: #ff4d6d;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 13px;
        display: inline-block;
        margin: 3px;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────
def check_api_health() -> bool:
    """Returns True if FastAPI backend is reachable."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=2)
        return resp.status_code == 200 and resp.json().get("model_loaded", False)
    except Exception:
        return False


def make_prediction(applicant_data: dict) -> dict | None:
    """Calls /predict endpoint. Returns response dict or None on failure."""
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json=applicant_data,
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"API error {resp.status_code}: {resp.json().get('detail','')}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Is the FastAPI server running?")
        return None


def probability_gauge(probability: float, decision: str) -> go.Figure:
    """
    Creates a Plotly gauge chart showing approval probability.
    This is the most visually impressive element of the dashboard.
    """
    color = "#00c896" if decision == "Approved" else "#ff4d6d"

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = probability * 100,
        delta = {"reference": THRESHOLD * 100,
                 "valueformat": ".1f",
                 "suffix": "%"},
        number= {"suffix": "%", "font": {"size": 48, "color": color}},
        gauge = {
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": "#a0a0a0", "tickfont": {"color": "#a0a0a0"}},
            "bar":  {"color": color, "thickness": 0.3},
            "bgcolor": "#1a1d27",
            "bordercolor": "#2a2d3a",
            "steps": [
                {"range": [0, 40],      "color": "rgba(255,77,109,0.15)"},
                {"range": [40, 65],     "color": "rgba(244,162,97,0.15)"},
                {"range": [65, 100],    "color": "rgba(0,200,150,0.15)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.85,
                "value": THRESHOLD * 100,
            },
        },
        title = {"text": "Approval Probability",
                 "font": {"size": 16, "color": "#a0a0a0"}},
        domain= {"x": [0, 1], "y": [0, 1]},
    ))

    fig.update_layout(
        paper_bgcolor = "#1a1d27",
        font          = {"color": "#e0e0e0"},
        height        = 280,
        margin        = dict(l=20, r=20, t=40, b=20),
    )
    return fig


def shap_waterfall_chart(factors: list[dict]) -> go.Figure:
    """Horizontal bar chart of SHAP feature contributions."""
    if not factors:
        return None

    features = [f["feature"].replace("_", " ").title() for f in factors]
    values   = [f["shap_value"] for f in factors]
    colors   = ["#00c896" if v > 0 else "#ff4d6d" for v in values]

    fig = go.Figure(go.Bar(
        x           = values,
        y           = features,
        orientation = "h",
        marker_color= colors,
        text        = [f"+{v:.4f}" if v > 0 else f"{v:.4f}" for v in values],
        textposition= "outside",
        textfont    = {"size": 11, "color": "#e0e0e0"},
    ))

    fig.add_vline(x=0, line_color="white", line_width=1, opacity=0.5)

    fig.update_layout(
        title       = "Feature Contributions (SHAP Values)",
        title_font  = {"size": 14, "color": "#e0e0e0"},
        paper_bgcolor= "#1a1d27",
        plot_bgcolor = "#1a1d27",
        font         = {"color": "#e0e0e0"},
        xaxis        = {"title": "SHAP value (+ = toward approval)",
                        "gridcolor": "#2a2d3a", "color": "#a0a0a0"},
        yaxis        = {"gridcolor": "#2a2d3a", "color": "#a0a0a0"},
        height       = 320,
        margin       = dict(l=10, r=80, t=50, b=40),
    )
    return fig


# ── Sidebar — applicant form ──────────────────────────────────────────────────
def render_sidebar() -> dict:
    """Renders the input form and returns applicant data dict."""
    st.sidebar.markdown("## 📋 Applicant Details")
    st.sidebar.markdown("---")

    with st.sidebar:
        st.markdown("**Personal Information**")
        gender    = st.selectbox("Gender",       ["Male", "Female"])
        married   = st.selectbox("Married",      ["Yes", "No"])
        dependents= st.selectbox("Dependents",   ["0", "1", "2", "3+"])
        education = st.selectbox("Education",    ["Graduate", "Not Graduate"])
        self_emp  = st.selectbox("Self Employed",["No", "Yes"])

        st.markdown("---")
        st.markdown("**Financial Information**")
        app_income  = st.number_input(
            "Applicant Income (₹/month)", min_value=0, max_value=500_000,
            value=5000, step=500,
            help="Monthly income of the primary applicant"
        )
        co_income   = st.number_input(
            "Co-applicant Income (₹/month)", min_value=0, max_value=500_000,
            value=0, step=500,
            help="Monthly income of co-applicant (0 if none)"
        )
        loan_amount = st.number_input(
            "Loan Amount (₹ thousands)", min_value=1, max_value=5000,
            value=150, step=10,
            help="Requested loan amount in thousands of rupees"
        )
        loan_term   = st.selectbox(
            "Loan Term (months)", [120, 180, 240, 300, 360, 480],
            index=4,
            help="Repayment period in months"
        )

        st.markdown("---")
        st.markdown("**Credit & Property**")
        credit_map    = {"Clean history (1)": 1.0,
                         "Has defaults (0)":  0.0,
                         "Unknown (-1)":     -1.0}
        credit_label  = st.selectbox("Credit History", list(credit_map.keys()))
        credit_history= credit_map[credit_label]

        property_area = st.selectbox("Property Area",
                                      ["Semiurban", "Urban", "Rural"])

        st.markdown("---")
        predict_btn = st.button("🔍 Predict Eligibility",
                                 type="primary",
                                 use_container_width=True)

    return {
        "data": {
            "Gender":           gender,
            "Married":          married,
            "Dependents":       dependents,
            "Education":        education,
            "Self_Employed":    self_emp,
            "ApplicantIncome":  float(app_income),
            "CoapplicantIncome":float(co_income),
            "LoanAmount":       float(loan_amount),
            "Loan_Amount_Term": float(loan_term),
            "Credit_History":   credit_history,
            "Property_Area":    property_area,
        },
        "predict": predict_btn,
    }


# ── Main dashboard ────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <h1 style='color:#e0e0e0; font-size:2.2em;'>🏦 Loan Eligibility Predictor</h1>
        <p style='color:#a0a0a0; font-size:1.1em;'>
            ML-powered loan assessment with explainable AI
        </p>
    </div>
    """, unsafe_allow_html=True)

    # API status banner
    api_ok = check_api_health()
    if api_ok:
        st.success("✅ API Connected | RandomForestClassifier | Threshold: 0.77")
    else:
        st.error("❌ API Offline — Start with: uvicorn api.main:app --port 8000")
        st.stop()

    # Sidebar form
    form = render_sidebar()

    # ── Default state — show instructions ────────────────────────────────────
    if "result" not in st.session_state and not form["predict"]:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <h2>📊</h2>
                <h3 style='color:#5b7fff'>ML Model</h3>
                <p style='color:#a0a0a0'>Random Forest with<br>21 features</p>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <h2>🎯</h2>
                <h3 style='color:#00c896'>AUC Score</h3>
                <p style='color:#a0a0a0'>0.8548 on<br>validation set</p>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class='metric-card'>
                <h2>🔍</h2>
                <h3 style='color:#f4a261'>Explainable AI</h3>
                <p style='color:#a0a0a0'>SHAP values for<br>every prediction</p>
            </div>""", unsafe_allow_html=True)

        st.info("👈 Fill in applicant details in the sidebar and click **Predict Eligibility**")
        return

    # ── Prediction triggered ──────────────────────────────────────────────────
    if form["predict"] and api_ok:
        with st.spinner("Analysing application..."):
            result = make_prediction(form["data"])

        if result:
            st.session_state["result"]  = result
            st.session_state["profile"] = form["data"]

    # ── Display results ───────────────────────────────────────────────────────
    if "result" in st.session_state:
        result  = st.session_state["result"]
        profile = st.session_state.get("profile", {})

        decision = result["decision"]
        prob     = result["approval_probability"]
        risk     = result["risk_category"]
        factors  = result["top_factors"]

        # Decision banner
        banner_class = "approved-banner" if decision == "Approved" else "rejected-banner"
        # In streamlit_app.py, after the decision banner:
        if result.get("warnings"):
            for w in result["warnings"]:
                st.warning(f"Model Warning: {w}")
        icon         = "✅" if decision == "Approved" else "❌"
        st.markdown(
            f"<div class='{banner_class}'>{icon} Loan {decision}</div>",
            unsafe_allow_html=True
        )
        

        # ── Row 1: gauge + applicant summary ─────────────────────────────────
        col_gauge, col_summary = st.columns([1, 1])

        with col_gauge:
            st.plotly_chart(
                probability_gauge(prob, decision),
                use_container_width=True
            )

        with col_summary:
            st.markdown("<div class='section-header'>Applicant Summary</div>",
                        unsafe_allow_html=True)

            summary_data = {
                "Field":  ["Income", "Co-income", "Loan Amount",
                            "Term", "Credit History", "Area"],
                "Value":  [
                    f"₹{profile.get('ApplicantIncome',0):,.0f}/mo",
                    f"₹{profile.get('CoapplicantIncome',0):,.0f}/mo",
                    f"₹{profile.get('LoanAmount',0)*1000:,.0f}",
                    f"{profile.get('Loan_Amount_Term',360):.0f} months",
                    {1.0:"Clean ✓", 0.0:"Defaults ✗",
                     -1.0:"Unknown ?"}.get(
                        profile.get('Credit_History',0), "—"),
                    profile.get("Property_Area", "—"),
                ]
            }
            st.dataframe(
                pd.DataFrame(summary_data),
                hide_index=True,
                use_container_width=True
            )

            # Risk badge
            risk_color = {"Low Risk":"#00c896",
                          "Medium Risk":"#f4a261",
                          "High Risk":"#ff4d6d"}.get(risk, "#a0a0a0")
            st.markdown(
                f"<div style='text-align:center; margin-top:10px;'>"
                f"<span style='background:{risk_color}22; border:1px solid {risk_color};"
                f"color:{risk_color}; padding:8px 24px; border-radius:20px;"
                f"font-weight:bold; font-size:16px;'>⚠ {risk}</span></div>",
                unsafe_allow_html=True
            )

            lat = result.get("processing_time_ms", 0)
            st.caption(f"⚡ Response time: {lat:.0f}ms")

        # ── Row 2: SHAP waterfall ─────────────────────────────────────────────
        st.markdown("<div class='section-header'>Why this decision?</div>",
                    unsafe_allow_html=True)

        chart = shap_waterfall_chart(factors)
        if chart:
            st.plotly_chart(chart, use_container_width=True)

        # Factor pills — plain-language summary
        pos = [f for f in factors if f["shap_value"] > 0]
        neg = [f for f in factors if f["shap_value"] < 0]

        col_pos, col_neg = st.columns(2)
        with col_pos:
            st.markdown("**Factors supporting approval:**")
            for f in pos:
                name = f["feature"].replace("_", " ").title()
                st.markdown(
                    f"<span class='factor-positive'>↑ {name} "
                    f"(+{f['shap_value']:.4f})</span>",
                    unsafe_allow_html=True
                )
        with col_neg:
            st.markdown("**Factors against approval:**")
            for f in neg:
                name = f["feature"].replace("_", " ").title()
                st.markdown(
                    f"<span class='factor-negative'>↓ {name} "
                    f"({f['shap_value']:.4f})</span>",
                    unsafe_allow_html=True
                )

        # ── Row 3: Key metrics ────────────────────────────────────────────────
        st.markdown("<div class='section-header'>Model Metrics</div>",
                    unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC-AUC",    "0.8548")
        m2.metric("Precision",  "0.8736")
        m3.metric("Recall",     "0.8941")
        m4.metric("Threshold",  "0.77")


if __name__ == "__main__":
    # Trigger first render with empty state
    if "result" not in st.session_state:
        st.session_state.clear()
    main()