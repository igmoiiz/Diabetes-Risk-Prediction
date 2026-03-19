import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import joblib
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main { background-color: #0f1117; }

    .risk-card {
        border-radius: 16px;
        padding: 28px 32px;
        margin: 12px 0;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .risk-high  { background: linear-gradient(135deg, #2d0a0a 0%, #1a0505 100%); border-color: #e74c3c55; }
    .risk-low   { background: linear-gradient(135deg, #0a2d12 0%, #051a09 100%); border-color: #2ecc7155; }
    .risk-pre   { background: linear-gradient(135deg, #2d2206 0%, #1a1503 100%); border-color: #f39c1255; }

    .risk-label-high { color: #e74c3c; font-size: 2.4rem; font-weight: 600; letter-spacing: -1px; }
    .risk-label-low  { color: #2ecc71; font-size: 2.4rem; font-weight: 600; letter-spacing: -1px; }
    .risk-label-pre  { color: #f39c12; font-size: 2.4rem; font-weight: 600; letter-spacing: -1px; }

    .confidence-text {
        font-family: 'DM Mono', monospace;
        font-size: 1.1rem;
        color: rgba(255,255,255,0.6);
        margin-top: 4px;
    }

    .prob-bar-container {
        background: rgba(255,255,255,0.06);
        border-radius: 8px;
        height: 10px;
        width: 100%;
        margin: 6px 0;
        overflow: hidden;
    }
    .prob-bar-high { background: #e74c3c; height: 10px; border-radius: 8px; transition: width 0.6s ease; }
    .prob-bar-low  { background: #2ecc71; height: 10px; border-radius: 8px; transition: width 0.6s ease; }
    .prob-bar-pre  { background: #f39c12; height: 10px; border-radius: 8px; transition: width 0.6s ease; }

    .factor-pill {
        display: inline-block;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.82rem;
        margin: 4px 4px 4px 0;
        font-family: 'DM Mono', monospace;
        color: rgba(255,255,255,0.75);
    }
    .factor-up   { border-color: #e74c3c55; color: #e74c3c; background: rgba(231,76,60,0.08); }
    .factor-down { border-color: #2ecc7155; color: #2ecc71; background: rgba(46,204,113,0.08); }

    .section-header {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: rgba(255,255,255,0.35);
        margin-bottom: 12px;
        margin-top: 24px;
    }

    .metric-mini {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-mini-value { font-size: 1.5rem; font-weight: 600; color: white; }
    .metric-mini-label { font-size: 0.75rem; color: rgba(255,255,255,0.4); margin-top: 2px; }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 0;
        font-size: 1rem;
        font-weight: 600;
        font-family: 'DM Sans', sans-serif;
        letter-spacing: 0.3px;
        cursor: pointer;
        transition: opacity 0.2s;
        margin-top: 8px;
    }
    .stButton > button:hover { opacity: 0.88; }

    div[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    .warning-box {
        background: rgba(243,156,18,0.08);
        border: 1px solid rgba(243,156,18,0.3);
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 0.82rem;
        color: rgba(243,156,18,0.9);
        margin-top: 16px;
    }

    .borderline-box {
        background: rgba(243,156,18,0.06);
        border: 1px solid rgba(243,156,18,0.25);
        border-radius: 10px;
        padding: 10px 14px;
        font-size: 0.8rem;
        color: rgba(243,156,18,0.8);
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model         = joblib.load("models/best_model_final.pkl")
    scaler        = joblib.load("models/scaler.pkl")
    le_target     = joblib.load("models/label_encoder.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, scaler, le_target, feature_names

@st.cache_resource
def load_explainer(_model, _scaler, _feature_names):
    """Build SHAP explainer — cached so it only runs once."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from scipy import stats
    from sklearn.linear_model import LogisticRegression

    # Rebuild training data to fit explainer
    df = pd.read_csv("dataset/diabetes_risk_dataset.csv")
    cols = [
        "ID","Age","Gender","BMI","BP","Fasting_Glucose_Level","Insulin_Level",
        "HbA1c_Level","Cholesterol_Level","Triglycerides_Level","Physical_Activity_Level",
        "Daily_Calorie_Intake","Sugar_Intake_Grams_Per_Day","Sleep_Hours","Stress_Level",
        "Family_History_Diabetes","Waist_Circumference_cm","Diabetes_Risk_Score",
        "Diabetes_Risk_Category",
    ]
    df.columns = cols
    df.drop(columns=["ID"], inplace=True)
    df.drop_duplicates(inplace=True)

    num_cols     = df.select_dtypes(include=["int64","float64"]).columns.tolist()
    outlier_cols = [c for c in num_cols if c != "Diabetes_Risk_Score"]
    zs           = np.abs(stats.zscore(df[outlier_cols]))
    df           = df[(zs < 3).all(axis=1)].reset_index(drop=True)
    df           = df.drop(columns=["Diabetes_Risk_Score","Insulin_Level","HbA1c_Level",
                                     "Sugar_Intake_Grams_Per_Day","Waist_Circumference_cm"])

    le = LabelEncoder()
    df["Gender"]                  = le.fit_transform(df["Gender"])
    df["Family_History_Diabetes"] = le.fit_transform(df["Family_History_Diabetes"])
    df = pd.get_dummies(df, columns=["Physical_Activity_Level"], drop_first=False)

    le_t = LabelEncoder()
    y    = le_t.fit_transform(df["Diabetes_Risk_Category"])
    X    = df.drop(columns=["Diabetes_Risk_Category"])

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=54, stratify=y)
    X_train_sc = _scaler.transform(X_train)
    X_train_df = pd.DataFrame(X_train_sc, columns=_feature_names)

    explainer = shap.LinearExplainer(_model, X_train_df)
    return explainer

try:
    model, scaler, le_target, feature_names = load_artifacts()
    CLASS_NAMES = le_target.classes_.tolist()
    explainer   = load_explainer(model, scaler, feature_names)
    artifacts_ok = True
except Exception as e:
    artifacts_ok = False
    load_error   = str(e)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def predict_patient(patient_dict):
    patient_df     = pd.DataFrame([patient_dict])[feature_names]
    patient_scaled = scaler.transform(patient_df)
    patient_shap   = pd.DataFrame(patient_scaled, columns=feature_names)

    pred_encoded = model.predict(patient_scaled)[0]
    probs        = model.predict_proba(patient_scaled)[0]
    pred_label   = le_target.inverse_transform([pred_encoded])[0]
    confidence   = probs.max() * 100

    sv_raw = explainer.shap_values(patient_shap)
    if isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 3:
        pred_sv = sv_raw[0, :, pred_encoded]
    elif isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 2:
        pred_sv = sv_raw[:, pred_encoded]
    else:
        pred_sv = sv_raw[pred_encoded][0]

    # Top factors with direction relative to predicted class
    contrib = pd.DataFrame({
        "feature": feature_names,
        "shap"   : pred_sv,
    }).assign(abs_shap=lambda d: d["shap"].abs()).sort_values("abs_shap", ascending=False)

    top_factors = []
    for _, row in contrib.head(5).iterrows():
        direction = "up" if row["shap"] > 0 else "down"
        top_factors.append({
            "feature"  : row["feature"].replace("_", " "),
            "direction": direction,
            "impact"   : abs(row["shap"]),
        })

    return {
        "prediction"    : pred_label,
        "confidence"    : round(confidence, 1),
        "probabilities" : {cls: round(p * 100, 1) for cls, p in zip(CLASS_NAMES, probs)},
        "top_factors"   : top_factors,
        "pred_encoded"  : pred_encoded,
    }


def make_shap_plot(patient_dict):
    patient_df     = pd.DataFrame([patient_dict])[feature_names]
    patient_scaled = scaler.transform(patient_df)
    patient_shap   = pd.DataFrame(patient_scaled, columns=feature_names)

    pred_encoded = model.predict(patient_scaled)[0]

    sv_raw = explainer.shap_values(patient_shap)
    if isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 3:
        pred_sv = sv_raw[0, :, pred_encoded]
        base_val = explainer.expected_value[pred_encoded]
    elif isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 2:
        pred_sv  = sv_raw[:, pred_encoded]
        base_val = explainer.expected_value[pred_encoded]
    else:
        pred_sv  = sv_raw[pred_encoded][0]
        base_val = explainer.expected_value[pred_encoded]

    shap_exp = shap.Explanation(
        values        = pred_sv,
        base_values   = base_val,
        data          = patient_shap.iloc[0].values,
        feature_names = [f.replace("_", " ") for f in feature_names],
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0f1117")
    shap.waterfall_plot(shap_exp, show=False, max_display=10)

    ax = plt.gca()
    ax.set_facecolor("#0f1117")
    ax.tick_params(colors="white", labelsize=9)
    ax.xaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor((1, 1, 1, 0.1))

    # Force ALL text objects to white — SHAP renders its own text internally
    for text_obj in fig.findobj(plt.Text):
        text_obj.set_color("white")

    plt.title(
        f"Why {CLASS_NAMES[pred_encoded]}?",
        color="white", fontsize=11, pad=10,
    )

    # Push left margin out so feature labels are not clipped
    plt.subplots_adjust(left=0.38, right=0.96, top=0.92, bottom=0.1)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — PATIENT INPUT FORM
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 Patient Input")
    st.markdown("<div style='font-size:0.78rem;color:rgba(255,255,255,0.4);margin-bottom:20px'>Enter patient values below</div>",
                unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Demographics</div>", unsafe_allow_html=True)
    age    = st.slider("Age",            min_value=18,  max_value=90,   value=45)
    gender = st.selectbox("Gender",      ["Male", "Female"])
    fhd    = st.selectbox("Family History of Diabetes", ["No", "Yes"])

    st.markdown("<div class='section-header'>Body Measurements</div>", unsafe_allow_html=True)
    bmi    = st.slider("BMI",            min_value=15.0, max_value=55.0, value=28.0, step=0.1)
    bp     = st.slider("Blood Pressure (mmHg)", min_value=80, max_value=210, value=130)

    st.markdown("<div class='section-header'>Blood Work</div>", unsafe_allow_html=True)
    glucose     = st.slider("Fasting Glucose (mg/dL)", min_value=50,  max_value=300, value=95)
    cholesterol = st.slider("Cholesterol (mg/dL)",     min_value=100, max_value=350, value=200)
    trigly      = st.slider("Triglycerides (mg/dL)",   min_value=40,  max_value=400, value=150)

    st.markdown("<div class='section-header'>Lifestyle</div>", unsafe_allow_html=True)
    activity  = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
    calories  = st.slider("Daily Calorie Intake",  min_value=1000, max_value=5000, value=2200)
    sleep     = st.slider("Sleep Hours",           min_value=3.0,  max_value=12.0, value=7.0, step=0.5)
    stress    = st.slider("Stress Level (1–10)",   min_value=1,    max_value=10,   value=5)

    st.markdown("")
    predict_btn = st.button("🔍  Predict Diabetes Risk")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# Diabetes Risk Predictor")
st.markdown(
    "<div style='color:rgba(255,255,255,0.45);font-size:0.92rem;margin-bottom:32px'>"
    "Logistic Regression · 94.02% accuracy · SHAP explainability"
    "</div>",
    unsafe_allow_html=True,
)

if not artifacts_ok:
    st.error(f"Could not load model files. Make sure you've run `explain.py` first.\n\nError: {load_error}")
    st.stop()

if not predict_btn:
    # Landing state
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='metric-mini'>
            <div class='metric-mini-value'>94.0%</div>
            <div class='metric-mini-label'>Model accuracy</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-mini'>
            <div class='metric-mini-value'>5,737</div>
            <div class='metric-mini-label'>Training patients</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-mini'>
            <div class='metric-mini-value'>3</div>
            <div class='metric-mini-label'>Risk categories</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈  Fill in patient details in the sidebar and click **Predict Diabetes Risk**.")

    st.markdown("### Risk Categories")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class='risk-card risk-high'>
            <div class='risk-label-high'>High Risk</div>
            <div style='color:rgba(255,255,255,0.5);font-size:0.85rem;margin-top:8px'>
                Likely diabetic. Immediate clinical follow-up recommended.
            </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='risk-card risk-pre'>
            <div class='risk-label-pre'>Prediabetes</div>
            <div style='color:rgba(255,255,255,0.5);font-size:0.85rem;margin-top:8px'>
                Borderline. Lifestyle changes can prevent progression.
            </div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='risk-card risk-low'>
            <div class='risk-label-low'>Low Risk</div>
            <div style='color:rgba(255,255,255,0.5);font-size:0.85rem;margin-top:8px'>
                No immediate concern. Routine monitoring advised.
            </div>
        </div>""", unsafe_allow_html=True)

else:
    # ── Build patient dict ────────────────────────────────────────────────────
    patient = {
        "Age"                             : age,
        "Gender"                          : 1 if gender == "Male" else 0,
        "BMI"                             : bmi,
        "BP"                              : bp,
        "Fasting_Glucose_Level"           : glucose,
        "Cholesterol_Level"               : cholesterol,
        "Triglycerides_Level"             : trigly,
        "Daily_Calorie_Intake"            : calories,
        "Sleep_Hours"                     : sleep,
        "Stress_Level"                    : stress,
        "Family_History_Diabetes"         : 1 if fhd == "Yes" else 0,
        "Physical_Activity_Level_High"    : 1 if activity == "High"     else 0,
        "Physical_Activity_Level_Low"     : 1 if activity == "Low"      else 0,
        "Physical_Activity_Level_Moderate": 1 if activity == "Moderate" else 0,
    }

    with st.spinner("Analysing patient data..."):
        result = predict_patient(patient)

    pred      = result["prediction"]
    conf      = result["confidence"]
    probs     = result["probabilities"]
    factors   = result["top_factors"]

    # ── Risk card color ───────────────────────────────────────────────────────
    card_class  = {"High Risk": "risk-high", "Low Risk": "risk-low", "Prediabetes": "risk-pre"}[pred]
    label_class = {"High Risk": "risk-label-high", "Low Risk": "risk-label-low", "Prediabetes": "risk-label-pre"}[pred]
    icon        = {"High Risk": "⚠️", "Low Risk": "✅", "Prediabetes": "🔶"}[pred]

    # ── Layout ────────────────────────────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        # Main result card
        st.markdown(f"""
        <div class='risk-card {card_class}'>
            <div style='font-size:0.72rem;letter-spacing:2px;text-transform:uppercase;
                        color:rgba(255,255,255,0.35);margin-bottom:10px'>PREDICTION</div>
            <div class='{label_class}'>{icon} {pred}</div>
            <div class='confidence-text'>Confidence: {conf}%</div>
        </div>""", unsafe_allow_html=True)

        # Borderline warning
        if conf < 70:
            st.markdown("""
            <div class='borderline-box'>
                ⚠️ Low confidence prediction — this patient sits near a decision boundary.
                Clinical judgement is especially important here.
            </div>""", unsafe_allow_html=True)

        # Probability breakdown
        st.markdown("<div class='section-header'>Probability Breakdown</div>", unsafe_allow_html=True)
        bar_colors = {"High Risk": "prob-bar-high", "Low Risk": "prob-bar-low", "Prediabetes": "prob-bar-pre"}
        for cls, prob in probs.items():
            is_pred = "↑ " if cls == pred else ""
            st.markdown(f"""
            <div style='margin-bottom:12px'>
                <div style='display:flex;justify-content:space-between;
                            font-size:0.82rem;color:rgba(255,255,255,0.6);margin-bottom:5px'>
                    <span>{is_pred}{cls}</span>
                    <span style='font-family:"DM Mono",monospace;font-weight:500;
                                 color:{"white" if cls == pred else "inherit"}'>{prob}%</span>
                </div>
                <div class='prob-bar-container'>
                    <div class='{bar_colors[cls]}' style='width:{prob}%'></div>
                </div>
            </div>""", unsafe_allow_html=True)

        # Top driving factors
        st.markdown("<div class='section-header'>Top Contributing Factors</div>", unsafe_allow_html=True)
        pills_html = ""
        for f in factors:
            pill_class = "factor-up" if f["direction"] == "up" else "factor-down"
            arrow      = "↑" if f["direction"] == "up" else "↓"
            pills_html += f"<span class='factor-pill {pill_class}'>{arrow} {f['feature']} ({f['impact']:.3f})</span>"
        st.markdown(f"<div>{pills_html}</div>", unsafe_allow_html=True)

        # Disclaimer
        st.markdown("""
        <div class='warning-box'>
            This tool is for research and educational purposes only.
            It is not a substitute for professional medical diagnosis or clinical judgement.
        </div>""", unsafe_allow_html=True)

    with right:
        # Patient summary
        st.markdown("<div class='section-header'>Patient Summary</div>", unsafe_allow_html=True)
        summary_cols = st.columns(2)
        metrics = [
            ("Age",        f"{age} yrs"),
            ("Gender",     gender),
            ("BMI",        f"{bmi:.1f}"),
            ("Blood Pressure", f"{bp} mmHg"),
            ("Fasting Glucose", f"{glucose} mg/dL"),
            ("Cholesterol", f"{cholesterol} mg/dL"),
            ("Triglycerides", f"{trigly} mg/dL"),
            ("Sleep",       f"{sleep} hrs"),
            ("Stress",      f"{stress}/10"),
            ("Activity",    activity),
            ("Family Hx",   fhd),
            ("Calories",    f"{calories} kcal"),
        ]
        for i, (label, val) in enumerate(metrics):
            with summary_cols[i % 2]:
                st.markdown(f"""
                <div class='metric-mini' style='margin-bottom:8px'>
                    <div class='metric-mini-value' style='font-size:1.1rem'>{val}</div>
                    <div class='metric-mini-label'>{label}</div>
                </div>""", unsafe_allow_html=True)

        # SHAP waterfall
        st.markdown("<div class='section-header' style='margin-top:20px'>SHAP Explanation</div>",
                    unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.78rem;color:rgba(255,255,255,0.35);margin-bottom:10px'>"
            "Red bars push toward this prediction. Blue bars push away from it."
            "</div>",
            unsafe_allow_html=True,
        )
        with st.spinner("Building explanation..."):
            shap_fig = make_shap_plot(patient)
        st.pyplot(shap_fig, use_container_width=True)
        plt.close(shap_fig)