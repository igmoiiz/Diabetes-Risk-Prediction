# ─────────────────────────────────────────────────────────────────────────────
# 1.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import warnings
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# 2.  GLOBAL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")

# Seaborn / Matplotlib defaults
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 100,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

# Pandas display
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.4f}".format)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  PLOT DIRECTORY STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
PLOTS_ROOT = "plots"

PLOT_DIRS = {
    # Pre-cleaning visualizations
    "dist_raw":         os.path.join(PLOTS_ROOT, "01_raw", "distributions"),
    "count_raw":        os.path.join(PLOTS_ROOT, "01_raw", "count_plots"),
    "box_before":       os.path.join(PLOTS_ROOT, "02_before_outlier_removal", "boxplots"),
    "grouped_before":   os.path.join(PLOTS_ROOT, "02_before_outlier_removal", "grouped_boxplots"),
    # Post-cleaning visualizations
    "box_after":        os.path.join(PLOTS_ROOT, "03_after_outlier_removal", "boxplots"),
    "grouped_after":    os.path.join(PLOTS_ROOT, "03_after_outlier_removal", "grouped_boxplots"),
    # Correlation & summary
    "correlation":      os.path.join(PLOTS_ROOT, "04_correlation"),
    # Category analysis
    "category":         os.path.join(PLOTS_ROOT, "05_category_analysis"),
    "models":           os.path.join(PLOTS_ROOT, "06_model_results")
}

for path in PLOT_DIRS.values():
    os.makedirs(path, exist_ok=True)

os.makedirs("models", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def save_plot(fig, directory: str, name: str, dpi: int = 150) -> None:
    """Save a matplotlib figure to *directory/name.png* and close it."""
    path = os.path.join(directory, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


def pretty(col: str) -> str:
    """Convert snake_case column name to a Title Case label."""
    return col.replace("_", " ").title()


def section(title: str) -> None:
    """Print a visually clear section header."""
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────────────────────
section("LOADING DATASET")
df = pd.read_csv("dataset/diabetes_risk_dataset.csv")

# Rename columns to standardised Title_Case names
COLUMN_NAMES = [
    "ID",
    "Age",
    "Gender",
    "BMI",
    "BP",
    "Fasting_Glucose_Level",
    "Insulin_Level",
    "HbA1c_Level",
    "Cholesterol_Level",          # fixed: Cholestrol -> Cholesterol
    "Triglycerides_Level",
    "Physical_Activity_Level",
    "Daily_Calorie_Intake",
    "Sugar_Intake_Grams_Per_Day",
    "Sleep_Hours",
    "Stress_Level",
    "Family_History_Diabetes",
    "Waist_Circumference_cm",
    "Diabetes_Risk_Score",
    "Diabetes_Risk_Category",
]
df.columns = COLUMN_NAMES

# Drop ID – not a feature
df.drop(columns=["ID"], inplace=True)

section("BASIC DATA QUALITY CHECK")
print(f"Shape              : {df.shape}")
print(f"\nMissing values per column:\n{df.isnull().sum()}")
print(f"\nDuplicate rows     : {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(f"Shape after dedup  : {df.shape}")

# Separate numeric vs. categorical columns
NUMERIC_COLS = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
CATEGORICAL_COLS = df.select_dtypes(include=["object", "str"]).columns.tolist()

# Key clinical features (excluding target columns for comparison plots)
KEY_FEATURES = [
    "Age",
    "BMI",
    "Fasting_Glucose_Level",
    "HbA1c_Level",
    "Waist_Circumference_cm",
    "Diabetes_Risk_Score",
    "Sugar_Intake_Grams_Per_Day",
    "BP",
    "Insulin_Level",
]
# Features used for outlier removal – exclude target/score columns
OUTLIER_COLS = [c for c in NUMERIC_COLS if c not in ("Diabetes_Risk_Score",)]

print(f"\nNumeric columns    : {NUMERIC_COLS}")
print(f"Categorical columns: {CATEGORICAL_COLS}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  DESCRIPTIVE STATISTICS  (EDA first, then visualize)
# ─────────────────────────────────────────────────────────────────────────────
section("DESCRIPTIVE STATISTICS")
print("\n--- DataFrame Info ---")
df.info()

print("\n--- Summary Statistics (Numeric) ---")
print(df[NUMERIC_COLS].describe().T.to_string())

print("\n--- Categorical Value Counts ---")
for col in CATEGORICAL_COLS:
    print(f"\n{col}:\n{df[col].value_counts()}")

section("IQR & QUANTILE ANALYSIS (Before Outlier Removal)")
print(f"\n{'Column':<30} {'Q1':>10} {'Q3':>10} {'IQR':>10} {'Outlier Low':>14} {'Outlier High':>14}")
print("-" * 82)
for col in NUMERIC_COLS:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    low = Q1 - 1.5 * IQR
    high = Q3 + 1.5 * IQR
    n_out = ((df[col] < low) | (df[col] > high)).sum()
    print(f"{col:<30} {Q1:>10.3f} {Q3:>10.3f} {IQR:>10.3f} {low:>14.3f} {high:>14.3f}   [{n_out} outliers]")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  RAW DATA VISUALIZATIONS  (before outlier removal)
# ─────────────────────────────────────────────────────────────────────────────
section("PLOTTING: Raw Distributions")

# 7a. Histograms + KDE for every numeric column
print("\n→ Numeric distribution plots …")
for col in NUMERIC_COLS:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(data=df, x=col, kde=True, color="steelblue", edgecolor="white", ax=ax)
    ax.set_title(f"Distribution of {pretty(col)}")
    ax.set_xlabel(pretty(col))
    ax.set_ylabel("Count")
    save_plot(fig, PLOT_DIRS["dist_raw"], f"dist_{col}")

# 7b. Count plots for categorical columns
print("\n→ Categorical count plots …")
for col in CATEGORICAL_COLS:
    fig, ax = plt.subplots(figsize=(8, 5))
    order = df[col].value_counts().index.tolist()
    sns.countplot(data=df, x=col, order=order, palette="Set2", ax=ax)
    ax.set_title(f"Count of {pretty(col)}")
    ax.set_xlabel(pretty(col))
    ax.set_ylabel("Count")
    for bar in ax.patches:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{int(bar.get_height())}",
            ha="center", va="bottom", fontsize=10,
        )
    save_plot(fig, PLOT_DIRS["count_raw"], f"count_{col}")

# 7c. Boxplots before outlier removal
print("\n→ Boxplots before outlier removal …")
for col in NUMERIC_COLS:
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=df, x=col, color="lightcoral", flierprops=dict(marker="o", markersize=3), ax=ax)
    ax.set_title(f"Boxplot of {pretty(col)}  [Before Outlier Removal]")
    ax.set_xlabel(pretty(col))
    save_plot(fig, PLOT_DIRS["box_before"], f"box_{col}_before")

# 7d. Grouped boxplots (key features vs. Risk Category) before outlier removal
print("\n→ Grouped boxplots by Risk Category (before outlier removal) …")
for col in KEY_FEATURES:
    if col not in df.columns:
        continue
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=df, x="Diabetes_Risk_Category", y=col,
        palette="Set2", flierprops=dict(marker="o", markersize=3), ax=ax,
    )
    ax.set_title(f"{pretty(col)} by Diabetes Risk Category  [Before Removal]")
    ax.set_xlabel("Diabetes Risk Category")
    ax.set_ylabel(pretty(col))
    save_plot(fig, PLOT_DIRS["grouped_before"], f"grouped_{col}_before")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  OUTLIER REMOVAL  (Z-Score on feature columns only)
# ─────────────────────────────────────────────────────────────────────────────
section("OUTLIER REMOVAL  (Z-Score | threshold = 3)")
rows_before = len(df)
z_scores = np.abs(stats.zscore(df[OUTLIER_COLS]))
mask = (z_scores < 3).all(axis=1)
df = df[mask].reset_index(drop=True)
rows_after = len(df)
print(f"Rows before : {rows_before}")
print(f"Rows after  : {rows_after}")
print(f"Rows removed: {rows_before - rows_after}  ({(rows_before - rows_after) / rows_before * 100:.2f}%)")

# Recompute numeric columns list after cleaning (same columns, but safe practice)
NUMERIC_COLS_CLEAN = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

# ─────────────────────────────────────────────────────────────────────────────
# 9.  POST-CLEANING VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
section("PLOTTING: After Outlier Removal")

# 9a. Boxplots after removal
print("\n→ Boxplots after outlier removal …")
for col in NUMERIC_COLS_CLEAN:
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=df, x=col, color="mediumseagreen", flierprops=dict(marker="o", markersize=3), ax=ax)
    ax.set_title(f"Boxplot of {pretty(col)}  [After Outlier Removal]")
    ax.set_xlabel(pretty(col))
    save_plot(fig, PLOT_DIRS["box_after"], f"box_{col}_after")

# 9b. Grouped boxplots after outlier removal
print("\n→ Grouped boxplots by Risk Category (after outlier removal) …")
for col in KEY_FEATURES:
    if col not in df.columns:
        continue
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=df, x="Diabetes_Risk_Category", y=col,
        palette="Set2", flierprops=dict(marker="o", markersize=3), ax=ax,
    )
    ax.set_title(f"{pretty(col)} by Diabetes Risk Category  [After Removal]")
    ax.set_xlabel("Diabetes Risk Category")
    ax.set_ylabel(pretty(col))
    save_plot(fig, PLOT_DIRS["grouped_after"], f"grouped_{col}_after")

# ─────────────────────────────────────────────────────────────────────────────
# 10.  CORRELATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
section("PLOTTING: Correlation Analysis")

# 10a. Full correlation heatmap
print("\n→ Correlation heatmap …")
corr_matrix = df[NUMERIC_COLS_CLEAN].corr()
fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))   # upper triangle mask
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f",
    cmap="coolwarm", center=0, linewidths=0.5,
    annot_kws={"size": 9}, ax=ax,
)
ax.set_title("Feature Correlation Heatmap (Lower Triangle)")
save_plot(fig, PLOT_DIRS["correlation"], "correlation_heatmap")

# 10b. Correlation with Diabetes_Risk_Score (bar chart)
print("\n→ Correlation with Diabetes Risk Score …")
if "Diabetes_Risk_Score" in corr_matrix.columns:
    risk_corr = (
        corr_matrix["Diabetes_Risk_Score"]
        .drop("Diabetes_Risk_Score")
        .sort_values(key=abs, ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in risk_corr.values]
    risk_corr.plot(kind="bar", color=colors, edgecolor="white", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Feature Correlation with Diabetes Risk Score")
    ax.set_ylabel("Pearson Correlation")
    ax.set_xlabel("Feature")
    ax.tick_params(axis="x", rotation=45)
    save_plot(fig, PLOT_DIRS["correlation"], "risk_score_correlation_bar")

# ─────────────────────────────────────────────────────────────────────────────
# 11.  CATEGORY-LEVEL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
section("PLOTTING: Category-Level Analysis")

# 11a. Pie chart – Diabetes Risk Category distribution
print("\n→ Risk category pie chart …")
cat_counts = df["Diabetes_Risk_Category"].value_counts()
fig, ax = plt.subplots(figsize=(8, 8))
wedge_props = {"edgecolor": "white", "linewidth": 2}
ax.pie(
    cat_counts, labels=cat_counts.index, autopct="%1.1f%%",
    startangle=140, wedgeprops=wedge_props,
    colors=sns.color_palette("Set2", len(cat_counts)),
)
ax.set_title("Diabetes Risk Category Distribution")
save_plot(fig, PLOT_DIRS["category"], "risk_category_pie")

# 11b. Mean feature values per Risk Category (heatmap)
print("\n→ Mean feature values per Risk Category …")
cat_means = df.groupby("Diabetes_Risk_Category")[NUMERIC_COLS_CLEAN].mean()
# Normalize per column for readability
cat_means_norm = (cat_means - cat_means.min()) / (cat_means.max() - cat_means.min() + 1e-9)
fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(
    cat_means_norm.T, annot=cat_means.T.round(1), fmt="g",
    cmap="YlOrRd", linewidths=0.5, annot_kws={"size": 8}, ax=ax,
)
ax.set_title("Normalised Mean Feature Values per Risk Category")
ax.set_xlabel("Risk Category")
ax.set_ylabel("Feature")
save_plot(fig, PLOT_DIRS["category"], "category_feature_heatmap")

# 11c. Violin plots — key features by Risk Category
print("\n→ Violin plots for key features by Risk Category …")
for col in KEY_FEATURES:
    if col not in df.columns:
        continue
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(
        data=df, x="Diabetes_Risk_Category", y=col,
        palette="Set3", inner="quartile", ax=ax,
    )
    ax.set_title(f"{pretty(col)} Distribution by Risk Category")
    ax.set_xlabel("Diabetes Risk Category")
    ax.set_ylabel(pretty(col))
    save_plot(fig, PLOT_DIRS["category"], f"violin_{col}_by_risk")

# 11d. Stacked bar – Family History vs. Risk Category
print("\n→ Family History vs. Risk Category stacked bar …")
if "Family_History_Diabetes" in df.columns:
    cross = pd.crosstab(df["Diabetes_Risk_Category"], df["Family_History_Diabetes"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    cross.plot(kind="bar", stacked=True, colormap="Set2", edgecolor="white", ax=ax)
    ax.set_title("Family History of Diabetes by Risk Category (%)")
    ax.set_xlabel("Diabetes Risk Category")
    ax.set_ylabel("Percentage (%)")
    ax.legend(title="Family History", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=45)
    save_plot(fig, PLOT_DIRS["category"], "family_history_vs_risk_category")

# ─────────────────────────────────────────────────────────────────────────────
# 12.  FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
section("FINAL DATASET SUMMARY (Post-Cleaning)")
print(f"\nShape              : {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDescriptive Stats:\n{df[NUMERIC_COLS_CLEAN].describe().T.to_string()}")

section("PLOT DIRECTORY SUMMARY")
for key, path in PLOT_DIRS.items():
    n = len([f for f in os.listdir(path) if f.endswith(".png")])
    print(f"  {path:<55}  →  {n} plots")

print("\n✅  EDA Complete. All plots saved under:", PLOTS_ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# 13. Model Preprocessing & Training Splits
# ─────────────────────────────────────────────────────────────────────────────
print("\n")

# Removing Un-necessary Columns and Features
df = df.drop(columns=["Insulin_Level", "HbA1c_Level", "Sugar_Intake_Grams_Per_Day", "Waist_Circumference_cm", "Diabetes_Risk_Score"])

print("Remaining Columns in Dataset:\n")
print(df.columns.tolist())

# Label Encoding
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
df["Family_History_Diabetes"] = le.fit_transform(df["Family_History_Diabetes"])

# Label Encoding the Target Variable
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(df["Diabetes_Risk_Category"])

print("\nTarget class mapping:")
for i, cls in enumerate(le_target.classes_):
    print(f"  {i} → {cls}")

# One-Hot Encoding for Physical Activity Level
df = pd.get_dummies(df, columns=["Physical_Activity_Level"], drop_first=False)

# Train and Test Variables
X = df.drop(columns=["Diabetes_Risk_Category"])
y = y_encoded

# Training and Testing Splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=54, stratify=y)

# Scaling After Splitting for no leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain size : {X_train.shape}")
print(f"Test size  : {X_test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 14. Model Trainings
# ─────────────────────────────────────────────────────────────────────────────
section("MODEL TRAINING")
 
models = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=54),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=54),
    "SVM"                 : SVC(random_state=54),
    "KNN"                 : KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes"         : GaussianNB(),
    "Decision Tree"       : DecisionTreeClassifier(random_state=54),
    "XGBoost"             : XGBClassifier(random_state=54, eval_metric="mlogloss"),
    "LightGBM"            : LGBMClassifier(random_state=54, verbose=-1),
    "CatBoost"            : CatBoostClassifier(random_state=54, verbose=0),
    "Neural Network"      : MLPClassifier(random_state=54, max_iter=500),
}
 
trained_models   = {}
predictions      = {}
accuracies       = {}
cv_scores        = {}
 
for name, model in models.items():
    print(f"\n  Training: {name} ...")
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
 
    trained_models[name] = model
    predictions[name]    = preds
    accuracies[name]     = accuracy_score(y_test, preds)
 
    # 15-fold cross-validation on training set
    cv = cross_val_score(model, X_train_scaled, y_train, cv=15, scoring="accuracy")
    cv_scores[name] = cv
 
    print(f"    Test Accuracy : {accuracies[name]:.4f}")
    print(f"    CV Accuracy   : {cv.mean():.4f} ± {cv.std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 15.  MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
section("Model Evaluation")

# Accuracy Summary Table
results_df = pd.DataFrame({
    "Model" : list(accuracies.keys()),
    "Test Accuracy" : list(accuracies.values()),
    "CV Mean"       : [cv_scores[m].mean() for m in accuracies],
    "CV Std"        : [cv_scores[m].std()  for m in accuracies],
}).sort_values("Test Accuracy", ascending=False).reset_index(drop=True)

print("\n--- Accuracy Summary ---")
print(results_df.to_string(index=False))

# Classification Reports
print("\n--- Classification Reports ---")
for name, preds in predictions.items():
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(classification_report(
        y_test, preds,
        target_names=le_target.classes_,   # shows readable class names
    ))
    print(f"  Saving CM for: '{name}'")

# Confusion Matrix - Saved as Plots
print("\n→ Saving confusion matrix plots ...")
for name, preds in predictions.items():
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=le_target.classes_,
        yticklabels=le_target.classes_,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    save_plot(fig, PLOT_DIRS["models"], f"cm_{name.replace(' ', '_')}")

# Accuracy Comparison as Bar Chart
print("\n→ Model accuracy comparison chart ...")
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(
    results_df["Model"],
    results_df["Test Accuracy"],
    color="steelblue", edgecolor="white",
)
ax.errorbar(
    results_df["CV Mean"], results_df["Model"],
    xerr=results_df["CV Std"],
    fmt="o", color="orange", capsize=4, label="CV Mean ± Std",
)
ax.set_xlabel("Accuracy")
ax.set_title("Model Comparison — Test Accuracy vs CV Accuracy")
ax.set_xlim(0, 1)
ax.axvline(0.5, color="red", linewidth=0.8, linestyle="--", label="50% baseline")
ax.legend()
for bar, val in zip(bars, results_df["Test Accuracy"]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9)
safe_name = name.replace(' ', '_').replace('/', '_')
save_plot(fig, PLOT_DIRS["models"], f"cm_{safe_name}")

# ─────────────────────────────────────────────────────────────────────────────
# 16.  SAVE BEST MODEL & PIPELINE ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
section("SAVING BEST MODEL")
 
best_model_name = results_df.iloc[0]["Model"]
best_model      = trained_models[best_model_name]
best_accuracy   = results_df.iloc[0]["Test Accuracy"]
 
print(f"\nBest model : {best_model_name}")
print(f"Accuracy   : {best_accuracy:.4f}")
 
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(scaler,     "models/scaler.pkl")
joblib.dump(le_target,  "models/label_encoder.pkl")
 
print("\n  [saved] models/best_model.pkl")
print("  [saved] models/scaler.pkl")
print("  [saved] models/label_encoder.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# 17.  EXAMPLE: PREDICT A NEW PATIENT
# ─────────────────────────────────────────────────────────────────────────────
section("EXAMPLE PREDICTION ON NEW PATIENT")
 
# Column order must match X after encoding
# X columns after get_dummies will be (in this order):
# Age, Gender, BMI, BP, Fasting_Glucose_Level, Cholesterol_Level,
# Triglycerides_Level, Daily_Calorie_Intake, Sleep_Hours, Stress_Level,
# Family_History_Diabetes,
# Physical_Activity_Level_High, Physical_Activity_Level_Low,
# Physical_Activity_Level_Moderate
 
feature_columns = X.columns.tolist()
 
new_patient = pd.DataFrame([{
    "Age"                            : 55,
    "Gender"                         : 1,      # 1 = Male
    "BMI"                            : 34.0,
    "BP"                             : 148,
    "Fasting_Glucose_Level"          : 118,
    "Cholesterol_Level"              : 220,
    "Triglycerides_Level"            : 180,
    "Daily_Calorie_Intake"           : 2600,
    "Sleep_Hours"                    : 6.5,
    "Stress_Level"                   : 7,
    "Family_History_Diabetes"        : 1,      # 1 = Yes
    "Physical_Activity_Level_High"   : 0,
    "Physical_Activity_Level_Low"    : 1,
    "Physical_Activity_Level_Moderate": 0,
}])[feature_columns]
 
new_patient_scaled = scaler.transform(new_patient)
pred_encoded       = best_model.predict(new_patient_scaled)
pred_label         = le_target.inverse_transform(pred_encoded)
 
print(f"\nPredicted risk category : {pred_label[0]}")
 
# ─────────────────────────────────────────────────────────────────────────────
# 18.  FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
section("PIPELINE COMPLETE")
print(f"\n  Best model     : {best_model_name}")
print(f"  Test accuracy  : {best_accuracy:.4f}")
print(f"  Models saved   : models/")
print(f"  Plots saved    : {PLOTS_ROOT}/")
print(f"\n✅  Full ML pipeline complete.")