import numpy as np
import pandas as pd
import os
import cloudpickle
import joblib

from scipy.stats import ks_2samp

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    classification_report,
)


REGRESSION_FOLDER = "regression_package"
CLASSIFICATION_FOLDER = "classification_package"


def import_classification_data():

    unseed_data_file = os.path.join(CLASSIFICATION_FOLDER, "X_test.pkl")
    unseed_target_path = os.path.join(CLASSIFICATION_FOLDER, "Y_test.pkl")

    print("Loading unseen data...")
    X_test = joblib.load(unseed_data_file)
    Y_test = joblib.load(unseed_target_path)

    return X_test, Y_test


def classification_prediction(X_test):
    figs_folder = "../artifacts"
    os.makedirs(figs_folder, exist_ok=True)

    model_path = os.path.join(CLASSIFICATION_FOLDER, "model.pkl")
    preprocessor_path = os.path.join(CLASSIFICATION_FOLDER, "pipeline.pkl")

    # Load
    with open(preprocessor_path, "rb") as f:
        preprocessor = cloudpickle.load(f)
    model = joblib.load(model_path)

    print("Applying preprocessing...")
    X_test_transformed = preprocessor.transform(X_test)

    print("Generating predictions...")
    y_pred = model.predict(X_test_transformed)

    return y_pred


def classification_metrics(Y_test, y_pred):

    # Evaluation metrics
    acc = accuracy_score(Y_test, y_pred)
    report_dict = classification_report(Y_test, y_pred, output_dict=True)

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Report: {report_dict}")


def import_regression_data():

    unseed_data_file = os.path.join(REGRESSION_FOLDER, "X_test.pkl")
    unseed_target_path = os.path.join(REGRESSION_FOLDER, "Y_test.pkl")

    print("Loading unseen data...")
    X_test = joblib.load(unseed_data_file)
    Y_test = joblib.load(unseed_target_path)

    return X_test, Y_test


def regression_prediction(X_test):
    figs_folder = "../artifacts"
    os.makedirs(figs_folder, exist_ok=True)

    model_path = os.path.join(REGRESSION_FOLDER, "model.pkl")
    preprocessor_path = os.path.join(REGRESSION_FOLDER, "pipeline.pkl")

    # Load
    with open(preprocessor_path, "rb") as f:
        preprocessor = cloudpickle.load(f)
    model = joblib.load(model_path)

    print("Applying preprocessing...")
    X_test_transformed = preprocessor.transform(X_test)

    print("Generating predictions...")
    y_pred = model.predict(X_test_transformed)

    return y_pred


def regression_metrics(Y_test, y_pred):

    # Evaluation metrics
    r2 = r2_score(Y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
    mae = mean_absolute_error(Y_test, y_pred)

    print(f"Test R²: {r2:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")


def evaluate_drift(task_folder, X_test):

    reference_df_path = os.path.join(task_folder, "X_rest.pkl")
    reference_df = joblib.load(reference_df_path)

    psi_results = {}
    ks_results = {}

    for col in X_test.select_dtypes(include=[np.number]).columns:
        ref = reference_df[col].dropna()
        new = X_test[col].dropna()

        # PSI
        bins = pd.qcut(ref, q=10, duplicates="drop")
        ref_dist = bins.value_counts(normalize=True)
        new_bins = pd.cut(new, bins=bins.cat.categories)
        new_dist = new_bins.value_counts(normalize=True)

        # Align distributions to avoid NaNs
        ref_dist, new_dist = ref_dist.align(new_dist, fill_value=1e-6)

        psi = sum((new_dist - ref_dist) * np.log(new_dist / ref_dist))
        psi_results[col] = psi

        # KS
        ks_stat, ks_p = ks_2samp(ref, new)
        ks_results[col] = {"ks_stat": ks_stat, "p_value": ks_p}

    drift_summary = pd.DataFrame(
        {
            "Feature": psi_results.keys(),
            "PSI": psi_results.values(),
            "KS_Stat": [ks_results[col]["ks_stat"] for col in psi_results.keys()],
            "KS_p_value": [ks_results[col]["p_value"] for col in psi_results.keys()],
        }
    )

    print(drift_summary)


def main():

    X_reg_test, Y_reg_test = import_regression_data()
    y_reg_pred = regression_prediction(X_reg_test)
    regression_metrics(Y_reg_test, y_reg_pred)

    evaluate_drift(REGRESSION_FOLDER, X_reg_test)

    X_cls_test, Y_cls_test = import_classification_data()
    y_cls_pred = classification_prediction(X_cls_test)
    classification_metrics(Y_cls_test, y_cls_pred)

    evaluate_drift(CLASSIFICATION_FOLDER, X_cls_test)


if __name__ == "__main__":
    main()
