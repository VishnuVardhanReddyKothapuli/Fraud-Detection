# 🛡️ Credit Card Fraud Detection System with XGBoost & Streamlit

A clean, beginner-friendly, and portfolio-ready end-to-end Machine Learning project that detects fraudulent credit card transactions using **XGBoost** and an interactive **Streamlit** web application.

This project is trained and evaluated on the benchmark **[Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)** collected by the Machine Learning Group of ULB (Université Libre de Bruxelles) and Worldline.

---

## 📌 1. Project Overview

Credit card fraud represents an extreme needle-in-a-haystack problem: out of hundreds of thousands of transactions, only a fraction of a percent are fraudulent. Relying solely on standard accuracy can lead to dangerous false senses of security.

This project implements the complete core ML workflow:
1. **Exploratory Data Analysis (EDA)** on 284,807 real-world European card transactions.
2. **Preprocessing & Robust Scaling** using Scikit-Learn `ColumnTransformer` (handling outliers in `Amount`).
3. **Stratified Partitioning** (80/20 split preserving the 0.172% fraud class ratio).
4. **Gradient Boosted Decision Trees** via **XGBoost Classifier**.
5. **Evaluation** focused on Precision, Recall, F1-Score, and ROC-AUC.
6. **Artifact Serialization** via Joblib.
7. **Interactive Web Application** built with Streamlit for real-time inference and risk profiling.

### End-to-End Workflow

```text
creditcard.csv (data/ - 284,807 transactions)
         ↓
fraud_detection.ipynb (notebooks/)
         ↓
Preprocessing (Drop 'Time', RobustScaler on 'Amount', passthrough V1-V28)
         ↓
Stratified Train/Test Split (80/20)
         ↓
XGBoost Model Training (XGBClassifier)
         ↓
Evaluation (Accuracy: 99.96%, Precision: 95.06%, Recall: 80.61%, ROC-AUC: 0.9806)
         ↓
Save Model Pipeline (model/fraud_detection_model.joblib)
         ↓
Streamlit Web App (app.py)
         ↓
User inputs features or selects 1-click test presets
         ↓
Prediction: ⚠️ Fraudulent vs. ✅ Legitimate + Probability & Risk Metric
```

---

## 📂 2. Project Directory Structure

```text
Fraud Detection/
│
├── data/
│   ├── creditcard.csv                # Complete Kaggle dataset (284,807 rows, 143.8 MB)
│   └── sample_presets.json           # Real test samples (normal & fraud) for quick demos
│
├── notebooks/
│   └── fraud_detection.ipynb         # End-to-end Jupyter Notebook with saved execution outputs & plots
│
├── model/
│   └── fraud_detection_model.joblib  # Serialized Scikit-Learn Pipeline + trained XGBoost model
│
├── app.py                            # Interactive Streamlit Web Application
├── requirements.txt                  # Minimal project dependencies
└── README.md                         # Project documentation
```

---

## 📊 3. Dataset Description

- **Source:** [Kaggle: Credit Card Fraud Detection (mlg-ulb/creditcardfraud)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Total Transactions:** 284,807
- **Fraudulent Transactions:** 492 (0.1727% fraud rate)
- **Legitimate Transactions:** 284,315 (99.8273%)

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `Time` | Float | Seconds elapsed between this transaction and the first transaction in the dataset (dropped during preprocessing) |
| `V1` to `V28` | Float | Numerical principal components obtained through PCA (confidentiality transformation) |
| `Amount` | Float | Transaction amount in USD/EUR |
| `Class` | Integer (Target) | Response variable (`1` for fraud, `0` for legitimate) |

---

## 💻 4. Technologies Used

- **Python 3.x**
- **Pandas**: Tabular data manipulation
- **NumPy**: Linear algebra and numerical operations
- **Scikit-Learn**: Stratified splitting, `RobustScaler`, `ColumnTransformer`, `Pipeline`, and classification metrics
- **XGBoost**: Gradient boosted decision tree classifier
- **Matplotlib & Seaborn**: Confusion matrix heatmap and ROC curve visualization
- **Joblib**: Model pipeline serialization
- **Streamlit**: Interactive user interface for real-time inference

---

## 🧠 5. Machine Learning Pipeline & Training

### Preprocessing & Scaling
- **`Time` Dropped**: The `Time` column is an arbitrary elapsed time counter spanning 48 hours and does not generalize to future transaction patterns.
- **Robust Scaling for `Amount`**: `Amount` contains extreme outliers (ranging from $0 to thousands of dollars). We use `RobustScaler` (based on the median and IQR), which is resilient to outliers.
- **PCA Features `V1`–`V28`**: Kept as-is, since they are already standardized outputs of PCA.

### Stratified Splitting
Because fraud is exceedingly rare (0.172%), standard random splitting risks uneven distribution of positive cases. We split the data using `stratify=y` with an 80/20 train/test ratio:
- **Training Set:** 227,845 transactions (394 frauds)
- **Testing Set:** 56,962 transactions (98 frauds)

### Model & Pipeline Architecture
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', RobustScaler(), ['Amount'])
    ],
    remainder='passthrough' # Passes V1..V28 through
)

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb_model)
])
model.fit(X_train, y_train)
```

---

## 📈 6. Model Evaluation Results

Evaluated on the unseen test set of **56,962 transactions**:

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Accuracy** | `99.96%` | Total correct predictions |
| **Precision (Fraud)** | `95.06%` | When flagged as fraud, 95% were truly fraud |
| **Recall (Fraud)** | `80.61%` | Caught >80% of all real fraud cases |
| **F1-Score (Fraud)**| `0.8722` | Harmonic balance between Precision & Recall |
| **ROC-AUC** | `0.9806` | Area under Receiver Operating Characteristic curve |

### Why Precision & Recall Matter More Than Accuracy:
In this Kaggle dataset, **99.827%** of transactions are legitimate.
- A naive dummy model that predicts every transaction is **Legitimate (0)** achieves **99.827% Accuracy**!
- However, that model catches **0 fraudulent transactions** ($Recall = 0.0\%$), leaving the institution completely vulnerable to theft.
- **Recall** measures how many actual fraud attempts are intercepted.
- **Precision** ensures normal customers do not experience false card declinations.

---

## 🚀 7. How to Run the Project Locally

### Step 1: Navigate to the Project Directory

```bash
cd "Fraud Detection"
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Jupyter Notebook (Optional)

To inspect EDA, step-by-step training, and evaluation plots:

```bash
jupyter notebook notebooks/fraud_detection.ipynb
```

*(Note: The notebook comes pre-executed with rendered outputs and visualizations.)*

### Step 4: Launch the Streamlit Web App

```bash
streamlit run app.py
```
*(or `python -m streamlit run app.py`)*

The web app will open automatically in your browser at `http://localhost:8501`.

---

## 🧪 8. Testing in the Streamlit Web App

The application includes built-in **1-click test presets**:
1. **🟢 Load Verified Legitimate Sample**:
   - Pulls a real normal transaction from the test set.
   - Result: `✅ Transaction Appears Legitimate` (Fraud Probability: `0.0006%`).
2. **🔴 Load Verified Fraudulent Sample**:
   - Pulls a real fraudulent transaction from the test set with characteristic negative anomalies in `V14`, `V10`, `V12`.
   - Result: `⚠️ Fraudulent Transaction Detected` (Fraud Probability: `99.93%`).
3. **Manual Adjustments**:
   - Adjust `Amount` and key anomaly indicators (`V14`, `V10`, `V12`, `V17`, `V4`, `V11`) or expand the full 28 PCA feature grid.

---

## 🎓 9. Key Interview Talking Points

1. **Class Imbalance Handling**: Explain the 0.172% fraud rate, why accuracy is completely deceptive, and how stratified sampling ensures reliable test evaluation.
2. **Scikit-Learn Pipeline**: Emphasize bundling `RobustScaler` on `Amount` with `XGBClassifier` into a unified `Pipeline` to prevent train-test data leakage and ensure seamless production inference in `app.py`.
3. **Domain Significance of PCA Indicators**: Note that features like `V14`, `V10`, and `V12` show extreme negative shifts during account takeover fraud, which the gradient boosted trees leverage effectively.
# Fraud-Detection
