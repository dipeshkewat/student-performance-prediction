# Student Performance Prediction Using Learning Analytics
*A Complete Big Data Analytics (BDA) College Mini-Project & Educational Machine Learning System*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Pytest-Passed-brightgreen.svg)](https://docs.pytest.org/)

---

## 1. Project Overview

Academic underperformance is frequently identified too late in a term for remedial pedagogical intervention. This project implements a **Student Performance Prediction** platform that analyzes historical academic milestones, attendance records, study behaviors, and institutional support systems to estimate a student's final grade (**G3** on the Portuguese 0–20 scale).

### Core Features
- **Data Ingestion & Integrity Validation:** Safe loading of CSV files with automatic delimiter detection (`,` and `;`), range checks (e.g., grades strictly within $[0, 20]$), duplicate detection, and missing value safety.
- **Leakage-Free Preprocessing:** A Scikit-learn `Pipeline` and `ColumnTransformer` fits imputers and scalers exclusively on the training partition to ensure zero data leakage.
- **Model Comparison & Rigorous Benchmarks:** Evaluates a baseline `DummyRegressor`, `LinearRegression`, and `RandomForestRegressor` via 5-fold cross-validation on the training set, followed by final evaluation on a 20% held-out test split.
- **Interactive Multi-Page Streamlit Dashboard:**
  - **Overview (`app.py`):** High-level KPI metrics, cohort distributions, and dataset provenance.
  - **Student Analytics (`1_Student_Analytics.py`):** Interactive Plotly visualizations exploring how study time, absences, and earlier-period grades influence the final grade.
  - **Predict Performance (`2_Predict_Performance.py`):** Real-time single student prediction form with risk factor analysis and a batch CSV processing workflow with export functionality.
  - **Model Evaluation (`3_Model_Evaluation.py`):** Actual vs. predicted scatter plots, residual diagnostics, algorithm benchmarks, and Gini feature importance attributions.
- **Automated Testing Suite:** 15 comprehensive unit and integration tests using `pytest` verifying data cleaning, schema validation, leak prevention, pipeline inference, and error handling.

---

## 2. Dataset Provenance & Features

The project is built on the **Student Performance Data Set** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance) (Cortez and Silva, 2008), focusing on secondary school mathematics records (`student-mat.csv`).

### Target Variable:
- **`G3`**: Final mathematics grade (integer scale: 0 to 20; passing threshold $\ge 10$).

### Input Features:
| Feature | Type | Range / Categories | Description |
| :--- | :--- | :--- | :--- |
| **`G1`** | Numeric | $0$ to $20$ | First-period academic grade |
| **`G2`** | Numeric | $0$ to $20$ | Second-period academic grade |
| **`studytime`** | Numeric | $1$ to $4$ | Weekly study time ($1$: $<2$h, $2$: $2$–$5$h, $3$: $5$–$10$h, $4$: $>10$h) |
| **`failures`** | Numeric | $0$ to $4$ | Number of previous class failures |
| **`absences`** | Numeric | $0$ to $93$ | Total number of class absences |
| **`schoolsup`** | Categorical | `yes`, `no` | Extra educational school support |
| **`famsup`** | Categorical | `yes`, `no` | Family educational support |
| **`activities`**| Categorical | `yes`, `no` | Extra-curricular activities participation |
| **`higher`** | Categorical | `yes`, `no` | Aspires to pursue higher education |
| **`internet`** | Categorical | `yes`, `no` | Internet access availability at home |

> **Important Timing Clarification:** Because the model leverages `G1` and `G2`, it predicts the final grade midway through the academic year when initial assessment milestones are available. It is not an admission or pre-enrollment tool.

---

## 3. Project Structure

```text
student-performance-prediction/
├── app.py                      # Main Streamlit entry point (Page A: Overview)
├── pages/
│   ├── 1_Student_Analytics.py  # Page B: Exploratory visualizations & cohort filters
│   ├── 2_Predict_Performance.py# Page C: Single prediction form & batch CSV processing
│   └── 3_Model_Evaluation.py   # Page D: Actual vs predicted, residuals, and metrics
├── data/
│   ├── raw/
│   │   └── student-mat.csv     # Authentic raw UCI dataset
│   └── processed/
│       ├── student_cleaned.csv # Verified, cleaned data artifact
│       └── train_test_split_info.json # Metadata on reproducible split
├── models/
│   └── best_pipeline.joblib    # Serialized champion model pipeline (ColumnTransformer + Model)
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # CSV loading with auto-delimiter detection & schema checks
│   ├── preprocessing.py        # Cleaning, range validation, leak prevention, ColumnTransformer
│   ├── train.py                # End-to-end training pipeline, CV, and model comparison
│   ├── evaluate.py             # MAE/RMSE/R2 computation and artifact serialization
│   ├── predict.py              # Single/batch inference and pedagogical risk analysis
│   └── dashboard_utils.py      # Common styling, cached loaders, sample CSV generator
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py   # Tests for data loading, delimiters, cleaning, and schemas
│   └── test_prediction.py      # Tests for inference, range bounds, batch CSV, and fallbacks
├── outputs/
│   ├── metrics.json            # Final test set metrics for champion pipeline
│   ├── model_comparison.json   # Benchmark metrics for Dummy, Linear, and Random Forest
│   ├── test_predictions.csv    # Actual, predicted, and residuals on held-out test set
│   └── feature_importance.json # Gini feature importance attribution
├── requirements.txt            # Python package dependencies
├── .gitignore                  # Git exclusions for Python & Streamlit
├── LICENSES_AND_DATA.md        # Dataset citation and academic terms
└── README.md                   # Comprehensive documentation
```

---

## 4. Setup and Execution Guide (Windows / PowerShell)

### Step 1: Clone or Navigate to the Project Root
```powershell
cd "f:\new bda"
```

### Step 2: Create and Activate Python Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell:
.venv\Scripts\Activate.ps1
```
*(If PowerShell displays an execution policy restriction, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Train Models and Generate Artifacts
Run the end-to-end training pipeline:
```powershell
python -m src.train
```
This loads raw data, validates features, splits into 80% train / 20% test sets, benchmarks Dummy, Linear Regression, and Random Forest models, selects the champion, and saves serialized artifacts to `models/` and `outputs/`.

### Step 5: Execute Automated Test Suite
Run `pytest` to verify all 15 tests:
```powershell
python -m pytest -v
```

### Step 6: Launch the Streamlit Interactive Dashboard
```powershell
python -m streamlit run app.py
```
*(Alternatively, if your Python Scripts folder is in your PATH, you can run `streamlit run app.py`).*
Open your browser to the local URL (typically `http://localhost:8501`).

---

## 5. Machine Learning Evaluation & Results

All models were evaluated on the exact same held-out test split (79 students, 20% of cohort):

| Model Candidate | Test MAE | Test RMSE | Held-Out Test $R^2$ | 5-Fold CV $R^2$ | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **DummyRegressor (Mean Baseline)** | $3.646$ | $4.550$ | $-0.010$ | $-0.009$ | Baseline |
| **LinearRegression** | $1.436$ | $2.194$ | $0.765$ | $0.834$ | Linear Candidate |
| **RandomForestRegressor** | **1.149** | **1.852** | **0.833** | **0.893** | **Champion Selected** |

### Key Findings:
1. **Strong Baseline Lift:** The champion `RandomForestRegressor` slashes prediction error by more than **$68\%$** compared to the naive mean baseline ($3.65 \to 1.15$ MAE).
2. **Dominant Predictors:** Feature importance analysis reveals that second-period grade (`G2`) and first-period grade (`G1`) account for over $75\%$ of predictive influence, followed by attendance rates (`absences`) and prior retentions (`failures`).

---

## 6. Big Data Scope & PySpark Extension Roadmap

- **Initial Scope:** The provided UCI dataset contains 395 students. In line with good engineering practices, this dataset is processed using Pandas and Scikit-Learn in memory without requiring unneeded distributed infrastructure (Hadoop/Spark clusters).
- **Modularity:** Preprocessing routines in `src/preprocessing.py` and `src/data_loader.py` are strictly separated from presentation logic.
- **Future PySpark Extension:** If deployed across statewide student databases (millions of student event streams), the Scikit-learn Pipeline can be replaced with `pyspark.ml.Pipeline` using `VectorAssembler` and `pyspark.ml.regression.RandomForestRegressor` while preserving identical dashboard interfaces and REST endpoints.

---

## 7. Responsible Interpretation & Ethical AI

- **Advisory Prototype Only:** Predictions are statistical indicators derived from historical correlation. They do not constitute guaranteed student capabilities or fatalistic destinies.
- **Early Remediation Focus:** The objective of this tool is early identification to allocate mentoring resources, tutoring programs, and study counseling.
- **Prohibition of High-Stakes Decisions:** This system must never be used to make consequential decisions such as student disqualification, academic tracking, or penalty assignments.
