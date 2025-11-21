import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from scipy.stats import ks_2samp
import os


# ---------------- CONFIG ----------------

MODEL_PATH = "artifacts/model.pkl"
PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"
REFERENCE_DATA_PATH = "artifacts/reference_data.csv"  # training data for drift checks
UNSEEN_DATA_PATH = "data/unseen.csv"
REPORT_PATH = "reports/evaluation_report.json"
PLOTS_DIR = "reports/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------- LOAD ----------------

print("Loading model and preprocessor...")
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

print("Loading unseen data...")
unseen_df = pd.read_csv(UNSEEN_DATA_PATH)
X_unseen = unseen_df.drop("target", axis=1)
y_true = unseen_df["target"]


# ---------------- PREPROCESS ----------------

print("Applying preprocessing...")
X_unseen_transformed = preprocessor.transform(X_unseen)


# ---------------- PREDICT ----------------

print("Generating predictions...")
y_pred = model.predict(X_unseen_transformed)
y_prob = model.predict_proba(X_unseen_transformed)[:, 1]  # for ROC/PR


# ---------------- METRICS ----------------

print("Computing metrics...")
metrics = {
    "accuracy": accuracy_score(y_true, y_pred),
    "f1": f1_score(y_true, y_pred),
    "roc_auc": roc_auc_score(y_true, y_prob),
}
print(metrics)


# ---------------- PLOTS ----------------

print("Generating plots...")
# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(f"{PLOTS_DIR}/confusion_matrix.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_prob)
plt.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.2f}")
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.savefig(f"{PLOTS_DIR}/roc_curve.png")
plt.close()

# Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_true, y_prob)
plt.plot(rec, prec)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.savefig(f"{PLOTS_DIR}/pr_curve.png")
plt.close()


# ---------------- DRIFT CHECK ----------------

print("Checking drift...")
reference_df = pd.read_csv(REFERENCE_DATA_PATH)
psi_results = {}
ks_results = {}

for col in X_unseen.columns:
    ref = reference_df[col]
    new = X_unseen[col]
    # PSI (simplified)
    bins = pd.qcut(ref, q=10, duplicates="drop")
    ref_dist = bins.value_counts(normalize=True)
    new_bins = pd.cut(new, bins=bins.cat.categories)
    new_dist = new_bins.value_counts(normalize=True)
    psi = sum((new_dist - ref_dist) * np.log(new_dist / ref_dist))
    psi_results[col] = psi
    # KS
    ks_stat, ks_p = ks_2samp(ref, new)
    ks_results[col] = {"ks_stat": ks_stat, "p_value": ks_p}


# ---------------- SAVE REPORT ----------------

report = {
    "metrics": metrics,
    "psi": psi_results,
    "ks": ks_results,
    "plots": {
        "confusion_matrix": f"{PLOTS_DIR}/confusion_matrix.png",
        "roc_curve": f"{PLOTS_DIR}/roc_curve.png",
        "pr_curve": f"{PLOTS_DIR}/pr_curve.png",
    },
}
with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=4)

print(f"Report saved to {REPORT_PATH}")
