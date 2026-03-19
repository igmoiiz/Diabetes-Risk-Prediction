# ─────────────────────────────────────────────────────────────────────────────
# 1.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def save_plot(fig, directory: str, name: str, dpi: int = 150) -> None:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


PLOTS_DIR = os.path.join("plots", "07_tuning_results")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  REBUILD DATA PIPELINE
#     (Exact same steps as main.py so X_train/X_test are identical)
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

# Outlier removal
NUMERIC_COLS = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
OUTLIER_COLS = [c for c in NUMERIC_COLS if c != "Diabetes_Risk_Score"]
z_scores = np.abs(stats.zscore(df[OUTLIER_COLS]))
df = df[(z_scores < 3).all(axis=1)].reset_index(drop=True)

# Drop leaky + removed columns
df = df.drop(columns=[
    "Diabetes_Risk_Score",
    "Insulin_Level",
    "HbA1c_Level",
    "Sugar_Intake_Grams_Per_Day",
    "Waist_Circumference_cm",
])

# Encode features
le = LabelEncoder()
df["Gender"]                  = le.fit_transform(df["Gender"])
df["Family_History_Diabetes"] = le.fit_transform(df["Family_History_Diabetes"])
df = pd.get_dummies(df, columns=["Physical_Activity_Level"], drop_first=False)

# Encode target
le_target = LabelEncoder()
y = le_target.fit_transform(df["Diabetes_Risk_Category"])
X = df.drop(columns=["Diabetes_Risk_Category"])

print(f"Classes : {list(enumerate(le_target.classes_))}")
print(f"X shape : {X.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=54, stratify=y
)

# Scale
scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)

print("✅  Data pipeline rebuilt.")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  BASELINE SCORES (untuned, for fair comparison)
# ─────────────────────────────────────────────────────────────────────────────
section("BASELINE SCORES (untuned defaults)")

baselines = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=54),
    "SVM"                 : SVC(random_state=54),
    "Neural Network"      : MLPClassifier(max_iter=500, random_state=54),
    "CatBoost"            : CatBoostClassifier(random_state=54, verbose=0),
    "XGBoost"             : XGBClassifier(random_state=54, eval_metric="mlogloss"),
    "LightGBM"            : LGBMClassifier(random_state=54, verbose=-1),
    "Random Forest"       : RandomForestClassifier(random_state=54),
}

baseline_scores = {}
for name, model in baselines.items():
    model.fit(X_train_sc, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_sc))
    baseline_scores[name] = acc
    print(f"  {name:<25} {acc:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  HYPERPARAMETER GRIDS
#
#     Strategy:
#     - Logistic Regression, SVM, Neural Network → GridSearchCV
#       (smaller search spaces, exhaustive is fine)
#     - Random Forest, XGBoost, LightGBM, CatBoost → RandomizedSearchCV
#       (large search spaces, random sampling is faster)
# ─────────────────────────────────────────────────────────────────────────────

# --- Logistic Regression ---
lr_grid = {
    "C"      : [0.01, 0.1, 1, 10, 100],      # regularization strength
    "solver" : ["lbfgs", "saga"],
    "penalty": ["l2"],
}

# --- SVM ---
svm_grid = {
    "C"      : [0.1, 1, 10, 50],             # margin hardness
    "kernel" : ["rbf", "linear"],
    "gamma"  : ["scale", "auto"],
}

# --- Neural Network ---
nn_grid = {
    "hidden_layer_sizes": [(64,), (128,), (64, 64), (128, 64)],
    "activation"        : ["relu", "tanh"],
    "alpha"             : [0.0001, 0.001, 0.01],   # L2 regularization
    "learning_rate"     : ["constant", "adaptive"],
}

# --- Random Forest ---
rf_grid = {
    "n_estimators"     : [100, 200, 300, 500],
    "max_depth"        : [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf" : [1, 2, 4],
    "max_features"     : ["sqrt", "log2"],
}

# --- XGBoost ---
xgb_grid = {
    "n_estimators"  : [100, 200, 300],
    "max_depth"     : [3, 5, 7, 9],
    "learning_rate" : [0.01, 0.05, 0.1, 0.2],
    "subsample"     : [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "gamma"         : [0, 0.1, 0.3],
}

# --- LightGBM ---
lgbm_grid = {
    "n_estimators"  : [100, 200, 300],
    "max_depth"     : [-1, 5, 10, 20],
    "learning_rate" : [0.01, 0.05, 0.1],
    "num_leaves"    : [31, 50, 100],
    "subsample"     : [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}

# --- CatBoost ---
cat_grid = {
    "iterations"   : [100, 200, 300],
    "depth"        : [4, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "l2_leaf_reg"  : [1, 3, 5, 7],
}

# ─────────────────────────────────────────────────────────────────────────────
# 6.  TUNING
# ─────────────────────────────────────────────────────────────────────────────
section("HYPERPARAMETER TUNING")

tuned_models  = {}
tuned_scores  = {}
best_params   = {}

CV_FOLDS    = 5
RANDOM_ITER = 50    # number of combinations RandomizedSearchCV tries
RANDOM_SEED = 54

# ── GridSearchCV models ───────────────────────────────────────────────────────

grid_configs = [
    (
        "Logistic Regression",
        LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        lr_grid,
    ),
    (
        "SVM",
        SVC(random_state=RANDOM_SEED),
        svm_grid,
    ),
    (
        "Neural Network",
        MLPClassifier(max_iter=500, random_state=RANDOM_SEED),
        nn_grid,
    ),
]

for name, estimator, grid in grid_configs:
    print(f"\n  GridSearchCV → {name} ...")
    t0 = time.time()

    search = GridSearchCV(
        estimator,
        grid,
        cv=CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train_sc, y_train)

    elapsed = time.time() - t0
    acc     = accuracy_score(y_test, search.best_estimator_.predict(X_test_sc))

    tuned_models[name] = search.best_estimator_
    tuned_scores[name] = acc
    best_params[name]  = search.best_params_

    print(f"    Best params   : {search.best_params_}")
    print(f"    CV best score : {search.best_score_:.4f}")
    print(f"    Test accuracy : {acc:.4f}  (was {baseline_scores[name]:.4f})")
    print(f"    Time taken    : {elapsed:.1f}s")

# ── RandomizedSearchCV models ─────────────────────────────────────────────────

rand_configs = [
    (
        "Random Forest",
        RandomForestClassifier(random_state=RANDOM_SEED),
        rf_grid,
    ),
    (
        "XGBoost",
        XGBClassifier(random_state=RANDOM_SEED, eval_metric="mlogloss"),
        xgb_grid,
    ),
    (
        "LightGBM",
        LGBMClassifier(random_state=RANDOM_SEED, verbose=-1),
        lgbm_grid,
    ),
    (
        "CatBoost",
        CatBoostClassifier(random_state=RANDOM_SEED, verbose=0),
        cat_grid,
    ),
]

for name, estimator, grid in rand_configs:
    print(f"\n  RandomizedSearchCV → {name} ...")
    t0 = time.time()

    search = RandomizedSearchCV(
        estimator,
        grid,
        n_iter=RANDOM_ITER,
        cv=CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=0,
    )
    search.fit(X_train_sc, y_train)

    elapsed = time.time() - t0
    acc     = accuracy_score(y_test, search.best_estimator_.predict(X_test_sc))

    tuned_models[name] = search.best_estimator_
    tuned_scores[name] = acc
    best_params[name]  = search.best_params_

    print(f"    Best params   : {search.best_params_}")
    print(f"    CV best score : {search.best_score_:.4f}")
    print(f"    Test accuracy : {acc:.4f}  (was {baseline_scores[name]:.4f})")
    print(f"    Time taken    : {elapsed:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  RESULTS COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
section("TUNING RESULTS — BEFORE vs AFTER")

comparison = pd.DataFrame({
    "Model"   : list(tuned_scores.keys()),
    "Before"  : [baseline_scores[m] for m in tuned_scores],
    "After"   : list(tuned_scores.values()),
}).assign(Gain=lambda d: (d["After"] - d["Before"]).round(4))
comparison = comparison.sort_values("After", ascending=False).reset_index(drop=True)

print(f"\n{'Model':<25} {'Before':>8} {'After':>8} {'Gain':>8}")
print("-" * 55)
for _, row in comparison.iterrows():
    gain_str = f"+{row.Gain:.4f}" if row.Gain >= 0 else f"{row.Gain:.4f}"
    print(f"  {row.Model:<23} {row.Before:>8.4f} {row.After:>8.4f} {gain_str:>8}")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  CLASSIFICATION REPORTS FOR TUNED MODELS
# ─────────────────────────────────────────────────────────────────────────────
section("CLASSIFICATION REPORTS (Tuned Models)")

for name, model in tuned_models.items():
    preds = model.predict(X_test_sc)
    print(f"\n{'─'*50}")
    print(f"  {name}  (tuned)")
    print(f"{'─'*50}")
    print(classification_report(y_test, preds, target_names=le_target.classes_))

# ─────────────────────────────────────────────────────────────────────────────
# 9.  PLOTS
# ─────────────────────────────────────────────────────────────────────────────
section("SAVING PLOTS")

# --- Before vs After grouped bar chart ---
fig, ax = plt.subplots(figsize=(13, 6))
x      = np.arange(len(comparison))
width  = 0.35

bars1 = ax.bar(x - width/2, comparison["Before"], width,
               label="Before tuning", color="steelblue",  edgecolor="white")
bars2 = ax.bar(x + width/2, comparison["After"],  width,
               label="After tuning",  color="mediumseagreen", edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(comparison["Model"], rotation=30, ha="right")
ax.set_ylabel("Test Accuracy")
ax.set_title("Hyperparameter Tuning — Before vs After")
ax.set_ylim(0.7, 1.0)
ax.legend()
ax.axhline(comparison["Before"].max(), color="steelblue",
           linewidth=0.8, linestyle="--", alpha=0.5)
ax.axhline(comparison["After"].max(),  color="mediumseagreen",
           linewidth=0.8, linestyle="--", alpha=0.5)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

save_plot(fig, PLOTS_DIR, "before_vs_after_tuning")

# --- Confusion matrices for tuned models ---
print("\n→ Saving confusion matrix plots (tuned) ...")
for name, model in tuned_models.items():
    preds     = model.predict(X_test_sc)
    cm        = confusion_matrix(y_test, preds)
    safe_name = name.replace(" ", "_")
    fig, ax   = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=le_target.classes_,
        yticklabels=le_target.classes_,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {name}  (tuned)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    save_plot(fig, PLOTS_DIR, f"cm_tuned_{safe_name}")

# ─────────────────────────────────────────────────────────────────────────────
# 10.  SAVE BEST TUNED MODEL
# ─────────────────────────────────────────────────────────────────────────────
section("SAVING BEST TUNED MODEL")

best_name  = comparison.iloc[0]["Model"]
best_model = tuned_models[best_name]
best_acc   = comparison.iloc[0]["After"]

print(f"\n  Best tuned model : {best_name}")
print(f"  Test accuracy    : {best_acc:.4f}")

joblib.dump(best_model, "models/best_model_tuned.pkl")
joblib.dump(scaler,     "models/scaler.pkl")
joblib.dump(le_target,  "models/label_encoder.pkl")

print("\n  [saved] models/best_model_tuned.pkl")
print("  [saved] models/scaler.pkl")
print("  [saved] models/label_encoder.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# 11.  BEST PARAMS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
section("BEST PARAMETERS SUMMARY")
for name, params in best_params.items():
    print(f"\n  {name}:")
    for k, v in params.items():
        print(f"    {k:<25} : {v}")

print(f"\n✅  Tuning complete. Best model: {best_name} @ {best_acc:.4f}")