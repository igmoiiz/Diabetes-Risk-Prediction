# ─────────────────────────────────────────────────────────────────────────────
# 1.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

warnings.filterwarnings("ignore")

PLOTS_DIR = os.path.join("plots", "08_shap")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def save_plot(fig, name: str, dpi: int = 150) -> None:
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  REBUILD DATA PIPELINE  (identical to main.py and tune.py)
# ─────────────────────────────────────────────────────────────────────────────
section("REBUILDING DATA PIPELINE")

df = pd.read_csv("dataset/diabetes_risk_dataset.csv")

COLUMN_NAMES = [
    "ID", "Age", "Gender", "BMI", "BP", "Fasting_Glucose_Level",
    "Insulin_Level", "HbA1c_Level", "Cholesterol_Level", "Triglycerides_Level",
    "Physical_Activity_Level", "Daily_Calorie_Intake", "Sugar_Intake_Grams_Per_Day",
    "Sleep_Hours", "Stress_Level", "Family_History_Diabetes",
    "Waist_Circumference_cm", "Diabetes_Risk_Score", "Diabetes_Risk_Category",
]
df.columns = COLUMN_NAMES
df.drop(columns=["ID"], inplace=True)
df.drop_duplicates(inplace=True)

NUMERIC_COLS = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
OUTLIER_COLS = [c for c in NUMERIC_COLS if c != "Diabetes_Risk_Score"]
z_scores     = np.abs(stats.zscore(df[OUTLIER_COLS]))
df           = df[(z_scores < 3).all(axis=1)].reset_index(drop=True)

df = df.drop(columns=[
    "Diabetes_Risk_Score", "Insulin_Level", "HbA1c_Level",
    "Sugar_Intake_Grams_Per_Day", "Waist_Circumference_cm",
])

le = LabelEncoder()
df["Gender"]                  = le.fit_transform(df["Gender"])
df["Family_History_Diabetes"] = le.fit_transform(df["Family_History_Diabetes"])
df = pd.get_dummies(df, columns=["Physical_Activity_Level"], drop_first=False)

le_target = LabelEncoder()
y         = le_target.fit_transform(df["Diabetes_Risk_Category"])
X         = df.drop(columns=["Diabetes_Risk_Category"])

FEATURE_NAMES = X.columns.tolist()
CLASS_NAMES   = le_target.classes_.tolist()   # ['High Risk', 'Low Risk', 'Prediabetes']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=54, stratify=y
)

scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)

# Keep unscaled version for readable SHAP plots
X_test_df   = pd.DataFrame(X_test_sc, columns=FEATURE_NAMES)
X_train_df  = pd.DataFrame(X_train_sc, columns=FEATURE_NAMES)

print(f"Classes : {list(enumerate(CLASS_NAMES))}")
print(f"X shape : {X.shape}")
print("✅  Pipeline ready.")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  TRAIN LOGISTIC REGRESSION WITH BEST TUNED PARAMS
#     (from tune.py: C=100, solver=saga, penalty=l2)
#     We use LR instead of SVM because:
#     - predict_proba() is natively calibrated in LR
#     - SVM requires Platt scaling which gives poorly calibrated probabilities
#     - Both achieved identical accuracy (94.13%)
# ─────────────────────────────────────────────────────────────────────────────
section("TRAINING LOGISTIC REGRESSION (best tuned params)")

model = LogisticRegression(
    C=100,
    solver="saga",
    penalty="l2",
    max_iter=1000,
    random_state=54,
)
model.fit(X_train_sc, y_train)

acc = accuracy_score(y_test, model.predict(X_test_sc))
print(f"\nTest accuracy : {acc:.4f}")
print(f"Classes       : {CLASS_NAMES}")

# Save this as the production model
joblib.dump(model,     "models/best_model_final.pkl")
joblib.dump(scaler,    "models/scaler.pkl")
joblib.dump(le_target, "models/label_encoder.pkl")
joblib.dump(FEATURE_NAMES, "models/feature_names.pkl")
print("\n  [saved] models/best_model_final.pkl  (LR, calibrated probabilities)")
print("  [saved] models/scaler.pkl")
print("  [saved] models/label_encoder.pkl")
print("  [saved] models/feature_names.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  CONFIDENCE SCORES — HOW THEY WORK
# ─────────────────────────────────────────────────────────────────────────────
section("CONFIDENCE SCORES — DEMO")

# predict()       → just the class label  e.g. 'High Risk'
# predict_proba() → probability for EACH class e.g. [0.82, 0.05, 0.13]
#                   sums to 1.0 across all 3 classes

sample        = X_test_sc[:5]
preds         = model.predict(sample)
probabilities = model.predict_proba(sample)   # shape: (5, 3)

print(f"\n{'Patient':<10} {'Prediction':<15} {'High Risk':>12} {'Low Risk':>12} {'Prediabetes':>12} {'Confidence':>12}")
print("-" * 75)
for i, (pred, probs) in enumerate(zip(preds, probabilities)):
    label      = le_target.inverse_transform([pred])[0]
    confidence = probs.max() * 100     # highest probability = confidence
    hr, lr, pd_ = probs * 100
    print(f"  {i+1:<8} {label:<15} {hr:>11.1f}% {lr:>11.1f}% {pd_:>11.1f}% {confidence:>11.1f}%")

print("\nInterpretation:")
print("  Confidence = probability of the predicted class")
print("  e.g. 'High Risk 92.3%' means the model is 92.3% sure this patient is High Risk")
print("  e.g. 'High Risk 51.2%' means borderline — treat with caution")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  CONFIDENCE DISTRIBUTION PLOT
#     Shows how confident the model is across all test patients
# ─────────────────────────────────────────────────────────────────────────────
section("PLOTTING: Confidence Score Distribution")

all_probs      = model.predict_proba(X_test_sc)
all_confidence = all_probs.max(axis=1) * 100
all_preds      = model.predict(X_test_sc)
all_labels     = le_target.inverse_transform(all_preds)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: overall confidence histogram
axes[0].hist(all_confidence, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
axes[0].axvline(all_confidence.mean(), color="red", linewidth=1.5,
                linestyle="--", label=f"Mean: {all_confidence.mean():.1f}%")
axes[0].axvline(90, color="orange", linewidth=1.2,
                linestyle=":", label="90% threshold")
axes[0].set_title("Model Confidence Distribution (All Test Patients)")
axes[0].set_xlabel("Confidence (%)")
axes[0].set_ylabel("Number of Patients")
axes[0].legend()

# Right: confidence by predicted class
colors = {"High Risk": "#e74c3c", "Low Risk": "#2ecc71", "Prediabetes": "#f39c12"}
for label in CLASS_NAMES:
    mask = all_labels == label
    axes[1].hist(
        all_confidence[mask], bins=20, alpha=0.6,
        label=f"{label} (n={mask.sum()})",
        color=colors[label], edgecolor="white",
    )
axes[1].set_title("Confidence by Predicted Risk Category")
axes[1].set_xlabel("Confidence (%)")
axes[1].set_ylabel("Number of Patients")
axes[1].legend()

plt.tight_layout()
save_plot(fig, "confidence_distribution")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  SHAP EXPLAINABILITY
#
#     SHAP (SHapley Additive exPlanations) answers:
#     "Which features pushed this prediction towards High Risk?"
#
#     For Logistic Regression we use LinearExplainer — fast and exact.
#     shap_values shape: (n_samples, n_features, n_classes)
#     Each value = how much that feature contributed to that class prediction
# ─────────────────────────────────────────────────────────────────────────────
section("SHAP ANALYSIS")

print("\nComputing SHAP values (LinearExplainer) ...")
explainer       = shap.LinearExplainer(model, X_train_df)
shap_values_raw = explainer.shap_values(X_test_df)

# Newer SHAP versions return shape (n_samples, n_features, n_classes)
# Older versions return a list of (n_samples, n_features) — one per class
# Normalize both into a consistent list of 3 arrays: [arr_class0, arr_class1, arr_class2]
if isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
    # (n_samples, n_features, n_classes) → split along last axis
    shap_values = [shap_values_raw[:, :, i] for i in range(shap_values_raw.shape[2])]
elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 2:
    # (n_features, n_classes) — transposed single-sample edge case, tile to full test set
    shap_values_raw = shap_values_raw.T   # → (n_classes, n_features)
    shap_values = [
        np.tile(shap_values_raw[i], (X_test_df.shape[0], 1))
        for i in range(shap_values_raw.shape[0])
    ]
else:
    # Already a list — old SHAP behavior
    shap_values = shap_values_raw

print(f"  SHAP values shape per class : {shap_values[0].shape}")
print(f"  Expected                    : ({X_test_df.shape[0]}, {len(FEATURE_NAMES)})")
print("  ✅  SHAP values computed.")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  PLOT 1 — Global Feature Importance (mean |SHAP|) per class
#     Answers: "Which features matter most for predicting each risk category?"
# ─────────────────────────────────────────────────────────────────────────────
section("PLOTTING: Global Feature Importance")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, (class_name, color) in enumerate(zip(CLASS_NAMES, ["#e74c3c", "#2ecc71", "#f39c12"])):
    mean_abs_shap = np.abs(shap_values[i]).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature"   : FEATURE_NAMES,
        "Importance": mean_abs_shap,
    }).sort_values("Importance", ascending=True)

    axes[i].barh(importance_df["Feature"], importance_df["Importance"],
                 color=color, alpha=0.8, edgecolor="white")
    axes[i].set_title(f"Feature Importance\n{class_name}")
    axes[i].set_xlabel("Mean |SHAP value|")
    axes[i].tick_params(axis="y", labelsize=9)

plt.suptitle("Global SHAP Feature Importance by Risk Category", fontsize=14, y=1.02)
plt.tight_layout()
save_plot(fig, "shap_global_importance_by_class")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  PLOT 2 — SHAP Summary Plot (beeswarm) for each class
#     Answers: "High feature value = higher or lower risk?"
# ─────────────────────────────────────────────────────────────────────────────
section("PLOTTING: SHAP Beeswarm Summary Plots")

for i, class_name in enumerate(CLASS_NAMES):
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values[i],
        X_test_df,
        feature_names=FEATURE_NAMES,
        show=False,
        plot_size=None,
    )
    plt.title(f"SHAP Summary — {class_name}", fontsize=13, pad=15)
    plt.tight_layout()
    save_plot(fig, f"shap_summary_{class_name.replace(' ', '_')}")

# ─────────────────────────────────────────────────────────────────────────────
# 10.  PLOT 3 — SHAP Waterfall for individual patients
#      Answers: "Why did THIS specific patient get this prediction?"
#      We show one example per class — the most confident prediction of each
# ─────────────────────────────────────────────────────────────────────────────
section("PLOTTING: Individual Patient Explanations (Waterfall)")

for class_idx, class_name in enumerate(CLASS_NAMES):
    # Find the most confident correct prediction for this class
    class_mask    = (all_preds == class_idx) & (y_test == class_idx)
    class_indices = np.where(class_mask)[0]

    if len(class_indices) == 0:
        print(f"  No correct predictions for {class_name}, skipping.")
        continue

    # Pick most confident example
    confidences      = all_probs[class_indices, class_idx]
    best_idx         = class_indices[np.argmax(confidences)]
    best_confidence  = confidences.max() * 100

    print(f"\n  {class_name} — most confident example:")
    print(f"    Test index  : {best_idx}")
    print(f"    Confidence  : {best_confidence:.1f}%")
    print(f"    Actual label: {CLASS_NAMES[y_test[best_idx]]}")

    # Build SHAP Explanation object for waterfall plot
    shap_exp = shap.Explanation(
        values      = shap_values[class_idx][best_idx],
        base_values = explainer.expected_value[class_idx],
        data        = X_test_df.iloc[best_idx].values,
        feature_names = FEATURE_NAMES,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap_exp, show=False)
    plt.title(
        f"Why '{class_name}'?  |  Confidence: {best_confidence:.1f}%  |  Patient #{best_idx}",
        fontsize=12, pad=12,
    )
    plt.tight_layout()
    save_plot(fig, f"shap_waterfall_{class_name.replace(' ', '_')}")

# ─────────────────────────────────────────────────────────────────────────────
# 11.  PREDICT FUNCTION — ready to use in your app
#      Returns: label, confidence %, and top 3 SHAP reasons
# ─────────────────────────────────────────────────────────────────────────────
section("PREDICTION FUNCTION — with confidence + explanation")

def predict_patient(patient_dict: dict) -> dict:
    """
    Takes a dict of patient features, returns prediction with
    confidence score and top contributing factors.

    Parameters
    ----------
    patient_dict : dict  — raw feature values (before scaling/encoding)
        Keys: Age, Gender (0/1), BMI, BP, Fasting_Glucose_Level,
              Cholesterol_Level, Triglycerides_Level, Daily_Calorie_Intake,
              Sleep_Hours, Stress_Level, Family_History_Diabetes (0/1),
              Physical_Activity_Level_High (0/1),
              Physical_Activity_Level_Low (0/1),
              Physical_Activity_Level_Moderate (0/1)

    Returns
    -------
    dict with keys:
        prediction  : str   — 'High Risk' / 'Low Risk' / 'Prediabetes'
        confidence  : float — 0.0 to 100.0
        probabilities: dict — all 3 class probabilities
        top_factors : list  — top 3 features that drove this prediction
    """
    # Build input dataframe in correct column order
    patient_df     = pd.DataFrame([patient_dict])[FEATURE_NAMES]
    patient_scaled = scaler.transform(patient_df)
    patient_shap   = pd.DataFrame(patient_scaled, columns=FEATURE_NAMES)

    # Predict
    pred_encoded  = model.predict(patient_scaled)[0]
    probs         = model.predict_proba(patient_scaled)[0]
    pred_label    = le_target.inverse_transform([pred_encoded])[0]
    confidence    = probs.max() * 100

    # SHAP explanation for predicted class
    # Handle both old (list) and new (ndarray) SHAP output formats
    sv_raw = explainer.shap_values(patient_shap)
    if isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 3:
        pred_sv = sv_raw[0, :, pred_encoded]
    elif isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 2:
        pred_sv = sv_raw[:, pred_encoded]
    else:
        pred_sv = sv_raw[pred_encoded][0]

    # Top 3 contributing features (by absolute SHAP value)
    contrib_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "shap"   : pred_sv,
    }).reindex(pd.Series(pred_sv).abs().sort_values(ascending=False).index)
    top_factors = []
    for _, row in contrib_df.head(3).iterrows():
        direction = "increases" if row["shap"] > 0 else "decreases"
        top_factors.append(f"{row['feature']} ({direction} risk, impact={row['shap']:.3f})")

    return {
        "prediction"    : pred_label,
        "confidence"    : round(confidence, 1),
        "probabilities" : {
            cls: round(p * 100, 1)
            for cls, p in zip(CLASS_NAMES, probs)
        },
        "top_factors"   : top_factors,
    }


# ── Demo: run predict_patient on 3 example patients ──────────────────────────
demo_patients = [
    {
        "label": "High-risk patient",
        "data": {
            "Age": 62, "Gender": 1, "BMI": 38.0, "BP": 165,
            "Fasting_Glucose_Level": 148, "Cholesterol_Level": 245,
            "Triglycerides_Level": 260, "Daily_Calorie_Intake": 3100,
            "Sleep_Hours": 5.0, "Stress_Level": 9,
            "Family_History_Diabetes": 1,
            "Physical_Activity_Level_High": 0,
            "Physical_Activity_Level_Low": 1,
            "Physical_Activity_Level_Moderate": 0,
        }
    },
    {
        "label": "Low-risk patient",
        "data": {
            "Age": 28, "Gender": 0, "BMI": 21.5, "BP": 115,
            "Fasting_Glucose_Level": 82, "Cholesterol_Level": 185,
            "Triglycerides_Level": 110, "Daily_Calorie_Intake": 1900,
            "Sleep_Hours": 8.0, "Stress_Level": 2,
            "Family_History_Diabetes": 0,
            "Physical_Activity_Level_High": 1,
            "Physical_Activity_Level_Low": 0,
            "Physical_Activity_Level_Moderate": 0,
        }
    },
    {
        "label": "Borderline patient",
        "data": {
            "Age": 45, "Gender": 1, "BMI": 29.5, "BP": 138,
            "Fasting_Glucose_Level": 105, "Cholesterol_Level": 215,
            "Triglycerides_Level": 175, "Daily_Calorie_Intake": 2400,
            "Sleep_Hours": 6.5, "Stress_Level": 6,
            "Family_History_Diabetes": 1,
            "Physical_Activity_Level_High": 0,
            "Physical_Activity_Level_Low": 0,
            "Physical_Activity_Level_Moderate": 1,
        }
    },
]

for patient in demo_patients:
    print(f"\n{'─'*55}")
    print(f"  {patient['label']}")
    print(f"{'─'*55}")
    result = predict_patient(patient["data"])
    print(f"  Prediction    : {result['prediction']}")
    print(f"  Confidence    : {result['confidence']}%")
    print(f"  Probabilities :")
    for cls, prob in result["probabilities"].items():
        bar = "█" * int(prob / 5)
        print(f"    {cls:<15} {prob:>5.1f}%  {bar}")
    print(f"  Top factors   :")
    for i, factor in enumerate(result["top_factors"], 1):
        print(f"    {i}. {factor}")

# ─────────────────────────────────────────────────────────────────────────────
# 12.  SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
section("COMPLETE")
print(f"\n  Model         : Logistic Regression (C=100, saga, l2)")
print(f"  Accuracy      : {acc:.4f}")
print(f"  Plots saved   : plots/08_shap/")
print(f"  Models saved  : models/")
print(f"\n  Files ready for deployment:")
print(f"    models/best_model_final.pkl")
print(f"    models/scaler.pkl")
print(f"    models/label_encoder.pkl")
print(f"    models/feature_names.pkl")
print(f"\n✅  SHAP analysis complete.")