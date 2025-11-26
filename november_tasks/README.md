
# Model Evaluation & Drift Analysis Pipeline

## Overview
This pipeline evaluates multiple regression and classification models, checks for data drift, and generates explainability plots (SHAP). 



## Project Structure

```
project-root/
│
├── data/                 # Processed datasets
├── qws1/                 # Raw datasets
├── Preprocess.ipynb      # Script for data pre-processing
├── Classification/       # Classification part with models & evaluation (tasks .1 and .2)
├── Regression/           # Regression part with models & evaluation (tasks .1 and .2)
├── artifacts/            # Figures of all parts of the project
└── outputs/              # Metrics, plots, drift reports
```

---

### Pipeline
  - **Feature Importance**: `permutation_importance`, `shap`
  - **Scalling**:`StandardScaler`, `MinMaxScaler`
  - **Enconding**: `BinaryEncoder`
  - **CV**: `RandomizedsearchCV`

- Features and Targets used:
  - **Regression**: Documentation, Availability, Successability, Reliability, Throughput, Best Practices to predict Response Time
  - **Classification**: Availability, Latency, Best Practices, Successability, Reliability, WsRF, Throughput, Documentation, Response Time, Compliance to predict Class

- Modeld used:
  - **Baseline Regression**: `LinearRegression`
  - **Baseline Classification**: `LogisticRegression`
  - **Regression**: `Ridge`, `Lasso`, `RANSACRegressor`, `DecisionTreeRegressor`, `RandomForestRegressor`, `SVR`
  - **Classification**: `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GaussianNB`, `SVC`, `KNeighborsClassifier`, `XGBClassifier`

- Metrics used:
  - **Regression**: `RMSE`, `MAE`, `R²`, `R1`
  - **Classification**: `Accuracy`, `F1`, `AUC`, `Precision/Recall`


---
## Project Features

### Prepare data
- Check nulls and dtypes
- Correct outliers
- Encoding categorical columns
- Plot correlation matrix

### Train and Test
- Define baseline model
- Try dimensionality reduction (via UMAP)
- Perform cross-validation and define best model
- Use best model to test data
- Save entire pipeline, model, metrics and plots

### Plots
- Residual plots for regression.
- ROC & PR curves for classification.
- Confusion matrix (classification only).

### Drift Checks on Test data
- **Feature distribution comparison** vs reference dataset.
- **PSI (Population Stability Index)** and **KS (Kolmogorov-Smirnov)** metrics computed.

## Run 

```
uv venv .venv
uv sync
.venv/Scripts/activate

# Preprocess
python preprocessing/run_preprocessing.py --config config.yaml

# Train and test models
python main.py

```