# Customer Churn Prediction Application

End-to-end machine learning project that predicts whether an e-commerce customer is likely to **churn** (stop using the platform). The system covers the entire lifecycle — from raw data cleaning and model training to a production-ready REST API with a browser-based frontend.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Dataset — `data/E_Commerce_Dataset.xlsx`](#1-dataset--datae_commerce_datasetxlsx)
3. [ML Notebook — `churn_prediction.ipynb`](#2-ml-notebook--churn_predictionipynb)
4. [Model Artifacts — `models/`](#3-model-artifacts--models)
5. [Backend (FastAPI) — `backend/`](#4-backend-fastapi--backend)
6. [Frontend (HTML/JS) — `frontend/`](#5-frontend-htmljs--frontend)
7. [Removal of Legacy Flask Monolith](#6-removal-of-legacy-flask-monolith)
8. [Root-Level Files](#7-root-level-files)
9. [System Architecture Diagram](#8-system-architecture-diagram)
10. [Setup & Run](#setup--run)
11. [API Example](#api-example)

---

## Project Structure

```text
customer-churn-prediction-application/
├── data/
│   └── E_Commerce_Dataset.xlsx            # Raw source dataset (Excel)
├── churn_prediction.ipynb                  # Jupyter notebook — EDA + cleaning + training
├── models/                                # Model artifacts (preserved for backend working)
│   ├── churn_prediction_model.pkl         # Trained XGBoost model (pickle)
│   └── columns.json                      # Feature column order used during training
├── backend/                               # FastAPI prediction API
│   ├── __init__.py
│   ├── main.py                            # App entry-point, routes, CORS
│   ├── predictor.py                       # Model loading, preprocessing, inference
│   └── schemas.py                         # Pydantic request/response schemas
├── frontend/                              # Static HTML/JS/CSS client
│   ├── index.html                         # Prediction form UI
│   ├── server.py                          # Python static-file server (port 5500)
│   └── static/
│       ├── app.js                         # Form → API call → result rendering
│       ├── styles.css                     # Custom styles
│       └── img/
│           └── churn.jpg                  # Header illustration
├── requirements.txt                       # Dependencies for FastAPI stack
├── PROJECT_IMPLEMENTATION_GUIDE.md        # Interview-prep walkthrough
└── README.md                              # ← You are here
```

---

## 1. Dataset — `data/E_Commerce_Dataset.xlsx`

### Source

[E-commerce Customer Churn Analysis and Prediction (Kaggle)](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction/data)

### Format

- **File type:** Microsoft Excel (`.xlsx`)
- **Sheet name:** `E Comm`
- **Rows:** ~5,630 customer records
- **Columns:** 20 raw features + 1 target (`Churn`)

### Feature Dictionary

| Column | Type | Description |
|---|---|---|
| CustomerID | int | Unique identifier — **dropped before training** (no predictive value) |
| Churn | int (0/1) | **Target variable.** 1 = churned, 0 = retained |
| Tenure | int | Months the customer has been with the platform |
| PreferredLoginDevice | str | Device used most for login (Computer, Mobile Phone, Phone) |
| CityTier | int (1-3) | Tier classification of the customer's city |
| WarehouseToHome | float | Distance (in km) from warehouse to customer's home |
| PreferredPaymentMode | str | Most used payment method (CC, COD, UPI, E wallet, etc.) |
| Gender | str | Male / Female |
| HourSpendOnApp | float | Average hours spent on the app per day |
| NumberOfDeviceRegistered | int | Count of devices registered by the customer |
| PreferedOrderCat | str | Preferred order category (Fashion, Mobile, Grocery, etc.) |
| SatisfactionScore | int (1-5) | Customer satisfaction rating |
| MaritalStatus | str | Single / Married / Divorced |
| NumberOfAddress | int | Number of saved delivery addresses |
| Complain | int (0/1) | Whether the customer raised a complaint in the last month |
| OrderAmountHikeFromlastYear | float | Percentage increase in order amount vs. previous year |
| CouponUsed | float | Number of coupons used in the last month |
| OrderCount | float | Number of orders placed in the last month |
| DaySinceLastOrder | float | Days elapsed since the most recent order |
| CashbackAmount | float | Average cashback amount received |

### Problems Found in the Dataset

| Problem | Details | Impact if Ignored |
|---|---|---|
| **Missing values** | `Tenure`, `WarehouseToHome`, `HourSpendOnApp`, `CouponUsed`, `OrderCount`, `DaySinceLastOrder`, `OrderAmountHikeFromlastYear` all had null entries | Model training would crash or produce biased results |
| **Mixed data types** | Categorical columns (`Gender`, `MaritalStatus`, `PreferredLoginDevice`, `PreferredPaymentMode`, `PreferedOrderCat`) are raw strings — ML models need numeric input | Model cannot process string features directly |
| **Non-predictive column** | `CustomerID` is a unique identifier with no pattern — keeping it would inject noise | Overfitting on arbitrary IDs |
| **Class imbalance** | Churn class (label `1`) is significantly smaller than non-churn (label `0`) | Model biases toward majority class; poor recall on actual churners |
| **Duplicate records** | Checked via `df.duplicated().sum()` | Could inflate training set and bias evaluation metrics |

### How the Data Was Cleaned (Step-by-Step)

| Step | What Was Done | Library / Function Used | Why This Approach |
|---|---|---|---|
| **1. Load** | Read Excel file into pandas DataFrame | `pd.read_excel("data/E_Commerce_Dataset.xlsx", sheet_name="E Comm")` | `openpyxl` engine handles `.xlsx` natively; `sheet_name` parameter targets the correct sheet |
| **2. Drop CustomerID** | `df.drop(columns="CustomerID", inplace=True)` | `pandas.DataFrame.drop()` | Unique ID has zero predictive power and would cause overfitting |
| **3. Lowercase columns** | `df.columns = [col.lower() for col in df.columns]` | Python list comprehension | Standardizes naming for consistent access throughout the pipeline |
| **4. EDA — missing value check** | `df.isnull().sum()` | `pandas.DataFrame.isnull()` + `.sum()` | Quantifies exactly how many NaN entries exist per column |
| **5. EDA — duplicate check** | `df.duplicated().sum()` | `pandas.DataFrame.duplicated()` | Ensures no inflated records |
| **6. EDA — distribution plots** | Count plots for categorical columns (`≤ 20 unique values`), histograms for continuous columns | `seaborn.countplot()`, `seaborn.histplot()` | Visual inspection reveals skew, outliers, and class distribution |
| **7. Churn distribution per feature** | Hue-based count plots with `hue="churn"` | `seaborn.countplot(hue="churn")` | Shows which feature values correlate with higher churn rates |
| **8. Impute numeric missing values** | Fill NaN with column mean | `sklearn.impute.SimpleImputer(strategy='mean')` | Mean imputation is simple and works well when data is roughly symmetric (no extreme outliers) |
| **9. One-hot encode categoricals** | Convert string columns to binary columns | `pd.get_dummies(df[col], prefix=col)` | Creates columns like `gender_male`, `gender_female` — converts categories to numeric representation ML models can consume |
| **10. Iterative Imputer** | Advanced imputation using Random Forest to predict missing values from other features | `sklearn.impute.IterativeImputer(estimator=RandomForestRegressor(random_state=42))` | Captures inter-feature relationships — if `Tenure` and `OrderCount` are correlated, the imputer uses that relationship instead of just using the mean. **Why Random Forest?** Non-parametric, handles non-linear relationships, robust to outliers |
| **11. Enable experimental** | Required to use `IterativeImputer` (it's experimental in sklearn) | `from sklearn.experimental import enable_iterative_imputer` | Sklearn gates experimental features behind an explicit import |

### Why This Cleaning Strategy Over Alternatives

| Decision | Alternative Considered | Why Rejected |
|---|---|---|
| `SimpleImputer(mean)` for numerics | Median, mode, or drop rows | Dropping rows loses data; mean is appropriate for roughly normal distributions in this dataset |
| `IterativeImputer` with RandomForest | KNN Imputer, constant fill | KNN is slow on larger datasets; constant fill ignores feature relationships |
| `pd.get_dummies()` for encoding | `sklearn.OneHotEncoder` | `get_dummies()` is simpler for notebook-style EDA; result is identical for this use case |
| Drop `CustomerID` | Keep it | It's a unique sequential ID — keeping it guarantees overfitting |

---

## 2. ML Notebook — `churn_prediction.ipynb`

### Purpose

This Jupyter notebook is the **training pipeline**. It takes the raw Excel dataset, cleans it, engineers features, trains multiple models, selects the best one (XGBoost), and exports the model + column schema for deployment.

### Cell-by-Cell Walkthrough

#### Phase 1: Import Libraries (Cell 0–1)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import missingno as msno
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
```

| Library | Role in This Project | Why This Library |
|---|---|---|
| **numpy** | Array operations, zeros initialization for feature vectors | Industry standard for numerical computing in Python |
| **pandas** | DataFrame operations — load, clean, transform, export | De facto standard for tabular data manipulation |
| **matplotlib** | Base plotting engine | Required by seaborn; used for confusion matrix display |
| **seaborn** | Statistical visualizations (count plots, histograms, hue-based plots) | Higher-level API than matplotlib; better defaults for EDA |
| **plotly.express** | Interactive visualizations | Allows hover, zoom, pan — useful for exploring distributions |
| **missingno** | Missing data visualization (matrix/heatmap of nulls) | Instantly shows NaN patterns across the entire dataset |
| **scikit-learn** | ML pipeline: imputation, encoding, splitting, model training, metrics | The most mature and well-documented ML library in Python |
| **xgboost** | Gradient boosted trees classifier | State-of-the-art for tabular data; consistently wins Kaggle competitions |

#### Phase 2: Load & Explore Data (Cells 2–15)

| Cell | Code | What It Does |
|---|---|---|
| 3 | `pd.set_option('display.max_columns', None)` | Shows all columns when printing DataFrame (no truncation) |
| 4 | `df = pd.read_excel("data/E_Commerce_Dataset.xlsx", sheet_name="E Comm")` | **Input:** Raw Excel file → **Output:** pandas DataFrame with 20 columns |
| 5 | `df.head()` | Preview first 5 rows |
| 6 | `df.shape` | Returns `(5630, 20)` — confirms row/column count |
| 7-8 | `df.describe()`, `df.info()` | Summary stats (mean, std, min, max) and data types |
| 9 | `df.info()` | Shows non-null counts per column — reveals which columns have missing values |
| 10 | `df.isnull().sum()` | **Critical step** — exact count of NaN per column |
| 11 | `df.duplicated().sum()` | Checks for duplicate rows |
| 12 | `df.nunique()` | Counts unique values per column — helps classify categorical vs. continuous |
| 13 | `df.drop(columns="CustomerID", inplace=True)` | **Drops the ID column** — prevents overfitting on a meaningless sequential number |
| 14-15 | Lowercase column names | Normalizes column names (e.g., `Tenure` → `tenure`) for consistency |

#### Phase 3: Visualization / EDA (Cells 16–23)

| Cell | Visualization | Function | Insight Gained |
|---|---|---|---|
| 17 | Segment columns by unique count | Python loop + `nunique()` | Columns with ≤ 20 unique values → count plots; others → histograms |
| 19 | Count plots for categorical columns | `sns.countplot()` | Shows frequency distribution of categories like `Gender`, `CityTier` |
| 21 | Histograms for numerical columns | `sns.histplot(bins=25)` | Reveals distribution shape, skewness, outliers for `Tenure`, `CashbackAmount`, etc. |
| 23 | Churn distribution per feature | `sns.countplot(hue="churn")` | **Key insight:** Which feature values have disproportionately high churn rates |

#### Phase 4: Missing Value Imputation (Cells 24–27)

The `fill_missing_values()` function is the **most important data cleaning code** in the project:

```python
def fill_missing_values(df, random_state=None):
    # Step 1: Identify numeric and categorical columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Step 2: Impute numeric columns with mean
    numeric_imputer = SimpleImputer(strategy='mean')
    df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

    # Step 3: One-hot encode categorical columns
    for col in categorical_columns:
        encoded_cols = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df.drop(col, axis=1), encoded_cols], axis=1)

    # Step 4: IterativeImputer with RandomForest for remaining gaps
    rf_imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=random_state))
    df = pd.DataFrame(rf_imputer.fit_transform(df), columns=df.columns)

    return df
```

**Step-by-step processing:**

1. **Input:** DataFrame with mixed types and NaN values
2. `select_dtypes()` separates numeric vs. object (string) columns
3. `SimpleImputer(strategy='mean')` fills numeric NaNs with column mean — fast first-pass
4. `pd.get_dummies()` converts each categorical column into multiple binary columns (one-hot encoding). Example: `Gender` → `gender_male` (0/1) + `gender_female` (0/1)
5. `IterativeImputer` treats each column with missing values as a regression target and predicts the missing value using all other columns as features. Uses `RandomForestRegressor` as the estimator.
6. **Output:** Fully numeric DataFrame with zero NaN values

#### Phase 5: Train/Test Split + Imbalance Handling (Cells 28–32)

```python
X = df.drop(columns=["churn"])
y = df["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

- `train_test_split()` splits 80% training / 20% testing
- `random_state=0` ensures reproducibility

**SMOTE (Synthetic Minority Over-sampling Technique):**

```python
from imblearn.over_sampling import SMOTE
sm = SMOTE(sampling_strategy=1, random_state=1)
X_train_s, y_train_s = sm.fit_resample(X_train, y_train.ravel())
```

| Concept | Explanation |
|---|---|
| **Problem** | Churn class (1) has far fewer samples than non-churn (0) — ~16.8% vs. ~83.2% |
| **Why it matters** | Model learns to predict "Not Churn" for everything and still gets ~83% accuracy |
| **SMOTE solution** | Generates **synthetic** minority samples by interpolating between existing minority data points in feature space |
| **`sampling_strategy=1`** | Makes class 1 count equal to class 0 |
| **Why SMOTE over random oversampling** | Random oversampling duplicates existing rows → overfitting. SMOTE creates *new* points → better generalization |
| **Library** | `imblearn` (imbalanced-learn) — purpose-built for handling class imbalance |

#### Phase 6: Model Evaluation (Cells 33–36)

Eight classifiers are compared using **5-fold cross-validation**:

| Model | Library | Type |
|---|---|---|
| AdaBoostClassifier | sklearn.ensemble | Boosting ensemble |
| BaggingClassifier | sklearn.ensemble | Bagging ensemble |
| GradientBoostingClassifier | sklearn.ensemble | Gradient boosting |
| RandomForestClassifier | sklearn.ensemble | Bagging + random feature subsets |
| LogisticRegressionCV | sklearn.linear_model | Linear with built-in CV |
| RidgeClassifierCV | sklearn.linear_model | L2-regularized linear |
| KNeighborsClassifier | sklearn.neighbors | Instance-based (lazy learner) |
| **XGBClassifier** | xgboost | **Extreme gradient boosting** |

**Metrics evaluated:** Accuracy, Precision, Recall, F1-Score

```python
scoring = ['accuracy', 'precision', 'recall', 'f1']
cv_results = model_selection.cross_validate(model, X, y, cv=5, scoring=scoring)
```

- `cross_validate()` with `cv=5` splits data into 5 folds, trains on 4, tests on 1, rotates — prevents overfitting to a single split.

#### Phase 7: Final Model Selection (Cells 37–41)

**XGBoost was selected** as the final model based on cross-validation results.

```python
final_model = XGBClassifier()
final_model.fit(X_train, y_train)
```

**Why XGBoost over RandomForest (which was also tested in Cell 37):**

| Criterion | XGBoost | RandomForest |
|---|---|---|
| Learning approach | Sequential boosting (each tree fixes errors of previous) | Parallel bagging (independent trees, majority vote) |
| Overfitting control | Built-in L1/L2 regularization | Only via tree depth / sample limits |
| Handling imbalance | `scale_pos_weight` parameter | Less flexible |
| Performance | Generally higher accuracy on structured/tabular data | Strong but usually slightly lower |
| Speed | Optimized C++ backend | Pure Python/Cython |

**Feature importance analysis** (Cells 40–41):
- `get_score(importance_type="gain")` — average improvement in loss function when a feature is used for splitting
- `get_score(importance_type="weight")` — how many times a feature is used across all tree splits

#### Phase 8: Feature Selection (Cells 42–45)

Columns dropped for the final deployment model:

```python
cols_to_drop = [
    'preferredlogindevice_Computer', 'preferredlogindevice_Mobile Phone', 'preferredlogindevice_Phone',
    'preferredpaymentmode_CC', 'preferredpaymentmode_COD', 'preferredpaymentmode_Cash on Delivery',
    'preferredpaymentmode_Credit Card', 'preferredpaymentmode_Debit Card',
    'preferredpaymentmode_E wallet', 'preferredpaymentmode_UPI',
    'preferedordercat_Fashion', 'preferedordercat_Grocery',
    'preferedordercat_Laptop & Accessory', 'preferedordercat_Mobile',
    'preferedordercat_Mobile Phone', 'preferedordercat_Others'
]
```

**Why these were dropped:** Feature importance analysis showed `PreferredLoginDevice`, `PreferredPaymentMode`, and `PreferedOrderCat` contribute minimally to churn prediction. Removing them creates a simpler, faster model with negligible accuracy loss.

**15 features retained for deployment:**
`tenure`, `citytier`, `warehousetohome`, `hourspendonapp`, `numberofdeviceregistered`, `satisfactionscore`, `numberofaddress`, `complain`, `orderamounthikefromlastyear`, `couponused`, `ordercount`, `daysincelastorder`, `cashbackamount`, `gender_*`, `maritalstatus_*`

#### Phase 9: Final Training + Evaluation (Cells 47–52)

```python
X_test = X_test.values   # Convert DataFrame → NumPy array
X_train = X_train.values

final_model = XGBClassifier()
final_model.fit(X_train, y_train)
```

**Why `.values` conversion?** XGBoost works with NumPy arrays internally. Converting early avoids DataFrame overhead and ensures compatibility with `pickle` serialization for deployment.

- **Confusion matrix** (Cell 50) visualizes True Positives, True Negatives, False Positives, False Negatives
- **Predictions comparison** (Cell 52) creates a DataFrame of Actual vs. Predicted for manual inspection

#### Phase 10: Model Serialization (Cells 53–55)

```python
# Save trained model
pickle.dump(final_model, open('models/churn_prediction_model.pkl', 'wb'))

# Save column order
columns = {'data_columns': [col.lower() for col in X.columns]}
with open("models/columns.json", "w") as f:
    f.write(json.dumps(columns))
```

**Why pickle?** Standard Python serialization for scikit-learn/XGBoost models. Alternatives like `joblib` exist but pickle is simpler and universally supported.

**Why `columns.json`?** At inference time, the model expects features in the **exact same order** as during training. `columns.json` stores this order so the backend can reconstruct the feature vector correctly.

---

## 3. Model Artifacts — `models/`

### `churn_prediction_model.pkl`

| Property | Value |
|---|---|
| Algorithm | XGBClassifier (XGBoost) |
| Format | Python `pickle` binary |
| Size | ~285 KB |
| Trained on | 80% of 5,630 records (after cleaning + SMOTE) |
| Output | `predict_proba()` returns `[P(Not Churn), P(Churn)]` |

**How it's loaded at runtime:**
```python
with open(MODEL_PATH, "rb") as model_file:
    self.model = pickle.load(model_file)
```

### `columns.json`

```json
{
  "data_columns": [
    "tenure", "citytier", "warehousetohome", "hourspendonapp",
    "numberofdeviceregistered", "satisfactionscore", "numberofaddress",
    "complain", "orderamounthikefromlastyear", "couponused",
    "ordercount", "daysincelastorder", "cashbackamount",
    "gender_female", "gender_male",
    "maritalstatus_divorced", "maritalstatus_married", "maritalstatus_single"
  ]
}
```

**18 columns total:** 13 numeric features + 2 gender one-hot columns + 3 marital status one-hot columns.

**Why this file is critical:** If you pass features in wrong order (e.g., `cashbackamount` where `tenure` should be), the model will produce garbage predictions without any error.

---

## 4. Backend (FastAPI) — `backend/`

### File: `backend/__init__.py`

**Content:** Empty file (just a newline).
**Purpose:** Marks the `backend/` directory as a Python package so that `from backend.predictor import ...` works.

---

### File: `backend/schemas.py`

**Purpose:** Defines the **data contract** between the frontend and the API using Pydantic models.

**Input:** N/A — this file is imported by `main.py`; it defines data shapes, not processing logic.
**Output:** Two Pydantic model classes that FastAPI uses for automatic validation and documentation.

```python
from pydantic import BaseModel, Field

class ChurnPredictionRequest(BaseModel):
    tenure: int = Field(..., ge=0, description="Customer tenure in the organization")
    citytier: int = Field(..., ge=1, le=3, description="City tier (1, 2, or 3)")
    warehousetohome: int = Field(..., ge=0, description="Distance from warehouse to home")
    gender: str = Field(..., description="Customer gender (male/female)")
    hourspendonapp: int = Field(..., ge=0, description="Hours spent on the app")
    numberofdeviceregistered: int = Field(..., ge=0, description="Number of registered devices")
    satisfactionscore: int = Field(..., ge=1, le=5, description="Satisfaction score (1-5)")
    maritalstatus: str = Field(..., description="Marital status (single/married/divorced)")
    numberofaddress: int = Field(..., ge=0, description="Number of saved addresses")
    complain: int = Field(..., ge=0, le=1, description="Whether customer complained (0 or 1)")
    orderamounthikefromlastyear: int = Field(..., ge=0, description="Order amount hike %")
    couponused: int = Field(..., ge=0, description="Number of coupons used")
    ordercount: int = Field(..., ge=0, description="Number of orders")
    daysincelastorder: int = Field(..., ge=0, description="Days since last order")
    cashbackamount: float = Field(..., ge=0, description="Average cashback amount")

class ChurnPredictionResponse(BaseModel):
    prediction: str          # "Churn" or "Not Churn"
    churn_probability: float # 0.0 to 1.0
    threshold: float         # 0.4
```

**Key design decisions:**

| Feature | Implementation | Why |
|---|---|---|
| **Pydantic `Field(...)`** | The `...` means "required" — no default value | Every field must be provided; missing fields return HTTP 422 automatically |
| **`ge=0`, `le=3`** etc. | Range validators | Prevents nonsensical input (e.g., negative tenure, city tier of 99) before it reaches the model |
| **`str` for gender/maritalstatus** | Kept as string, converted later in `predictor.py` | Clean separation — schema validates shape, predictor handles encoding |
| **Why Pydantic over manual validation** | Pydantic auto-generates JSON schema, error messages, and OpenAPI docs | Eliminates hundreds of lines of manual `if/else` validation code |

---

### File: `backend/predictor.py`

**Purpose:** The **core ML inference engine**. Loads the trained model once at startup, accepts customer data dictionaries, preprocesses them to match training format, and returns predictions.

**Input:** A Python dictionary with 15 customer fields (from `ChurnPredictionRequest.model_dump()`).
**Output:** A tuple of `(prediction_label: str, churn_probability: float)`.

```python
import json, pickle
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_prediction_model.pkl"
COLUMNS_PATH = PROJECT_ROOT / "models" / "columns.json"

CHURN_THRESHOLD = 0.4
CATEGORICAL_FIELDS = {"gender", "maritalstatus"}
```

#### Constants Explained

| Constant | Value | Why |
|---|---|---|
| `PROJECT_ROOT` | Resolved path two levels up from `predictor.py` | Uses `Path(__file__).resolve().parent.parent` to work from any working directory |
| `CHURN_THRESHOLD` | `0.4` | Lower than default 0.5 to **prioritize catching potential churners** (higher recall at the cost of some precision). Business rationale: cost of missing a churner > cost of a false alarm |
| `CATEGORICAL_FIELDS` | `{"gender", "maritalstatus"}` | These two fields need one-hot encoding; all other fields are already numeric |

#### Class: `ChurnPredictor`

**`__init__(self)`** — Called once when the FastAPI app starts:

```python
def __init__(self):
    # Validate files exist
    if not MODEL_PATH.exists():
        raise FileNotFoundError(...)
    # Load model from pickle
    with MODEL_PATH.open("rb") as model_file:
        self.model = pickle.load(model_file)
    # Load column order
    with COLUMNS_PATH.open("r", encoding="utf-8") as columns_file:
        self.data_columns = json.load(columns_file)["data_columns"]
```

**Why load once at startup?** Loading a pickle file and deserializing a model takes ~50–200ms. Doing this per-request would add unacceptable latency. Loading once and reusing the `self.model` object is the standard pattern for ML serving.

**`_prepare_input_array(self, customer_data: dict)`** — Transforms raw input into the exact format the model expects:

```python
def _prepare_input_array(self, customer_data: dict) -> np.ndarray:
    # Step 1: Normalize string values to lowercase + underscores
    normalized = {}
    for key, value in customer_data.items():
        if isinstance(value, str):
            normalized[key] = value.lower().replace(" ", "_")
        else:
            normalized[key] = value

    # Step 2: Initialize zero array matching training column count
    input_array = np.zeros(len(self.data_columns))  # 18 zeros

    # Step 3: Fill numeric values
    for index, column in enumerate(self.data_columns):
        if column in normalized and column not in CATEGORICAL_FIELDS:
            input_array[index] = normalized[column]

    # Step 4: One-hot encode categorical values
    for key, value in normalized.items():
        if key in CATEGORICAL_FIELDS:
            one_hot_column = f"{key}_{value}"  # e.g., "gender_male"
            if one_hot_column in self.data_columns:
                input_array[self.data_columns.index(one_hot_column)] = 1

    return input_array
```

**Processing steps visualized:**

```
Input: {"tenure": 10, "gender": "Male", "maritalstatus": "Single", ...}
  ↓
Step 1 — Normalize: {"tenure": 10, "gender": "male", "maritalstatus": "single", ...}
  ↓
Step 2 — Zero array: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  ↓
Step 3 — Fill numerics: [10, 3, 15, 3, 2, 4, 2, 0, 12, 1, 5, 7, 150.5, 0, 0, 0, 0, 0]
  ↓
Step 4 — One-hot encode: [10, 3, 15, 3, 2, 4, 2, 0, 12, 1, 5, 7, 150.5, 0, 1, 0, 0, 1]
                                                                              ^  ^        ^
                                                                         female male  single
```

**`predict(self, customer_data: dict)`** — Main inference method:

```python
def predict(self, customer_data: dict) -> tuple[str, float]:
    input_array = self._prepare_input_array(customer_data)
    churn_probability = float(self.model.predict_proba([input_array])[0][1])
    churn_probability = round(churn_probability, 4)
    prediction = "Churn" if churn_probability > CHURN_THRESHOLD else "Not Churn"
    return prediction, churn_probability
```

| Code | Explanation |
|---|---|
| `predict_proba([input_array])` | Returns `[[P(class_0), P(class_1)]]` — probabilities for both classes |
| `[0][1]` | Extracts `P(class_1)` — the probability of churning |
| `round(..., 4)` | Limits to 4 decimal places for clean API output |
| `> CHURN_THRESHOLD (0.4)` | Converts continuous probability into a binary business decision |

---

### File: `backend/main.py`

**Purpose:** The **FastAPI application entry-point**. Defines HTTP routes, configures CORS, and wires everything together.

**Input:** HTTP requests from the frontend (or any client).
**Output:** JSON responses.

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.predictor import CHURN_THRESHOLD, ChurnPredictor
from backend.schemas import ChurnPredictionRequest, ChurnPredictionResponse

app = FastAPI(
    title="Customer Churn Prediction API",
    description="FastAPI backend for predicting customer churn using a trained ML model.",
    version="1.0.0",
)
```

#### CORS Middleware

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Why CORS is needed:** The frontend runs on `http://127.0.0.1:5500` and the backend on `http://127.0.0.1:8000`. Browsers block cross-origin requests by default. `CORSMiddleware` with `allow_origins=["*"]` permits the frontend to call the backend API.

#### Singleton Predictor

```python
predictor = ChurnPredictor()  # Loaded once at module level
```

This runs when the FastAPI app starts, loading the model and columns into memory once.

#### Routes

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | Returns `{"status": "ok"}` — used for health checks / uptime monitoring |
| `/api/predict` | POST | Accepts customer data, returns churn prediction |

**Predict endpoint flow:**

```python
@app.post("/api/predict", response_model=ChurnPredictionResponse)
def predict_churn(payload: ChurnPredictionRequest) -> ChurnPredictionResponse:
    try:
        prediction, churn_probability = predictor.predict(payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return ChurnPredictionResponse(
        prediction=prediction,
        churn_probability=churn_probability,
        threshold=CHURN_THRESHOLD,
    )
```

**Processing steps:**

1. FastAPI automatically deserializes the JSON body into a `ChurnPredictionRequest` object (validates all fields)
2. If validation fails → auto-returns HTTP 422 with detailed error messages
3. `payload.model_dump()` converts the Pydantic model back to a plain dict
4. `predictor.predict()` preprocesses + runs inference
5. Result wrapped in `ChurnPredictionResponse` → auto-serialized to JSON
6. If model crashes → caught by `except` → returns HTTP 500 with error detail

**Why FastAPI over Flask:**

| Aspect | FastAPI | Flask |
|---|---|---|
| Request validation | Automatic via Pydantic | Manual (`request.form`, `if/else`) |
| API docs | Auto-generated Swagger at `/docs` | Need `flask-swagger` extension |
| Performance | ASGI (async) — 2–3x faster | WSGI (synchronous) |
| Type hints | First-class support | Not integrated |
| Error messages | Structured JSON with field-level details | Manual formatting |

---

## 5. Frontend (HTML/JS) — `frontend/`

### File: `frontend/index.html`

**Purpose:** The **user-facing prediction form**. Renders a Bootstrap 4 styled page with 15 input fields matching the 15 features the model expects.

**Input:** User types/selects values in the form fields.
**Output:** Rendered HTML page; on submit, triggers JavaScript to call the API.

**Key sections:**

| Section | HTML Element | Purpose |
|---|---|---|
| Header image | `<img src="static/img/churn.jpg">` | Visual branding |
| Form fields (15) | `<input type="number">` and `<select>` | Collect customer data |
| Gender dropdown | `<select id="gender">` with `female`/`male` options | Ensures valid categorical input |
| Marital status dropdown | `<select id="maritalstatus">` with `divorced`/`married`/`single` | Ensures valid categorical input |
| Complain field | `<input type="number" min="0" max="1">` | Binary flag (0 or 1) |
| Cashback field | `<input type="number" step="0.01">` | Accepts decimal values |
| Predict button | `<button type="submit">` | Triggers form submission |
| Loading indicator | `<p id="loading-message" class="hidden">` | Shown during API call |
| Error display | `<p id="error-message" class="hidden">` | Shown if API fails |
| Result section | `<div id="result-section" class="hidden">` | Shows prediction + probability |
| Predict Again button | `<button id="predict-again">` | Resets UI for next prediction |

**Why Bootstrap 4:** Provides responsive grid layout, form styling, and button components with minimal custom CSS. The `col-md-3` / `col-md-4` grid classes create a clean 4-column and 3-column layout for the form fields.

**Why `type="number"` instead of `type="text"` (unlike the old Flask version):** Adds native browser validation — prevents users from entering letters in numeric fields. Also shows numeric keyboard on mobile devices.

---

### File: `frontend/static/app.js`

**Purpose:** The **client-side logic**. Handles form submission, builds the JSON payload, sends it to the FastAPI backend via `fetch()`, and renders the result.

**Input:** DOM events (form submit, button click).
**Output:** Updated DOM elements showing prediction results or errors.

#### Function-by-Function Breakdown

**`getFormPayload()`** — Builds the request JSON:

```javascript
function getFormPayload() {
    return {
        tenure: Number(document.getElementById("tenure").value),
        citytier: Number(document.getElementById("citytier").value),
        // ... (15 fields total)
        cashbackamount: Number(document.getElementById("cashbackamount").value),
    };
}
```

| Detail | Explanation |
|---|---|
| `Number(...)` | Converts string from input field to JavaScript number — matches Pydantic's `int`/`float` types |
| `document.getElementById(...)` | Gets the form element by its HTML `id` attribute |
| String fields (`gender`, `maritalstatus`) | Passed as-is (no `Number()` conversion) — the backend expects strings |

**`predictChurn(event)`** — Main async handler:

```javascript
async function predictChurn(event) {
    event.preventDefault();       // Stop default form POST
    clearError();                  // Hide previous errors
    loadingMessage.classList.remove("hidden");  // Show "Running prediction..."
    resultSection.classList.add("hidden");      // Hide previous result

    try {
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(getFormPayload()),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Prediction request failed.");

        predictionLabel.textContent = data.prediction;
        predictionLabel.className = data.prediction === "Churn" ? "churn" : "not-churn";
        probabilityValue.textContent = data.churn_probability;
        resultSection.classList.remove("hidden");
    } catch (error) {
        showError(`Unable to get prediction. Make sure the FastAPI backend is running...`);
    } finally {
        loadingMessage.classList.add("hidden");
    }
}
```

**Processing flow:**

```
Form Submit
    ↓
event.preventDefault()        — Stops page reload
    ↓
getFormPayload()              — Reads 15 form values → JS object
    ↓
JSON.stringify()              — Converts to JSON string
    ↓
fetch() POST to /api/predict  — Sends to backend
    ↓
await response.json()         — Parses JSON response
    ↓
Update DOM                    — Shows "Churn" (red) or "Not Churn" (green)
```

**Why `fetch()` over alternatives:**

| Option | Why Not Used |
|---|---|
| `fetch()` ✅ | Native browser API, no dependencies, supports async/await |
| `XMLHttpRequest` | Older, callback-based, more verbose |
| `axios` | Would require adding a dependency for a simple POST request |
| `jQuery.ajax()` | Would require jQuery as a dependency |

**Event listeners:**

```javascript
form.addEventListener("submit", predictChurn);
predictAgainButton.addEventListener("click", () => {
    resultSection.classList.add("hidden");
    clearError();
    window.scrollTo({ top: 0, behavior: "smooth" });
});
```

- Submit listener calls `predictChurn` on form submit
- "Predict Again" button hides the result, clears errors, and smooth-scrolls to top

---

### File: `frontend/static/styles.css`

**Purpose:** Custom CSS overrides on top of Bootstrap 4.

| Class | Effect | Why Needed |
|---|---|---|
| `body` | Light blue background (`#ddefff`), Arial font, padding | Brand consistency, readability |
| `form` | White card with rounded corners and box shadow | Visual grouping of form fields |
| `.churn` | Red text (`#c0392b`) | Immediately signals danger/risk |
| `.not-churn` | Green text (`#27ae60`) | Immediately signals safety |
| `.hidden` | `display: none` | Toggle visibility of loading, error, result sections |
| `.error-message` | Red bold text | Makes errors unmissable |
| `.loading` | Centered bold text | Loading state feedback |

---

### File: `frontend/server.py`

**Purpose:** A minimal Python HTTP server that serves the frontend files on `http://127.0.0.1:5500`.

```python
import http.server, socketserver
from pathlib import Path

PORT = 5500
FRONTEND_DIR = Path(__file__).resolve().parent

class FrontendHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), FrontendHandler) as httpd:
        print(f"Frontend running at http://127.0.0.1:{PORT}")
        httpd.serve_forever()
```

| Component | Purpose |
|---|---|
| `http.server.SimpleHTTPRequestHandler` | Python stdlib — serves static files (HTML, JS, CSS, images) |
| `directory=str(FRONTEND_DIR)` | Ensures files are served from the `frontend/` folder regardless of where you run the script from |
| `socketserver.TCPServer` | Creates a TCP socket listening on port 5500 |
| `serve_forever()` | Blocks and keeps the server running |

**Why a Python server instead of just opening the HTML file?**

| Approach | Problem |
|---|---|
| Double-click `index.html` | Opens with `file://` protocol — browsers block `fetch()` calls from `file://` due to CORS |
| Python HTTP server ✅ | Serves via `http://` — `fetch()` works correctly |
| Node.js `http-server` | Would work but adds a Node.js dependency to a Python project |

---

## 6. Removal of Legacy Flask Monolith

The legacy Flask-based monolith (`end_to_end_deployment/`) has been removed to streamline the codebase around the modern FastAPI backend and dedicated frontend.

### Preservation of Essential Working Files
Before removing the Flask monolith files (`app.py`, `Procfile`, `requirements`, `templates/`, and `static/`), the essential ML model artifacts were preserved and relocated to the project root:
- `models/churn_prediction_model.pkl` (Trained XGBoost model)
- `models/columns.json` (Feature schema & column ordering)

The FastAPI backend (`backend/predictor.py`) now directly references `models/` for seamless prediction inference.

### Key Architectural Differences (FastAPI vs Old Flask)

| Aspect | Old Flask (Removed) | New FastAPI + Frontend |
|---|---|---|
| Architecture | Monolith — one file did everything | Separated — API + UI are independent |
| UI rendering | Server-side Jinja2 templates | Client-side JavaScript |
| Data transfer | HTML form POST → server-side rendering | JSON REST API (`/api/predict`) |
| Model loading | Per-request (re-read pickle file each time) | Once at startup (singleton pattern) |
| Input validation | None (relied on `isdigit()` check) | Pydantic auto-validation with type & range constraints |
| Error handling | Minimal | Automatic HTTP 422 for validation, 500 for inference |

---

## 7. Root-Level Files

### `requirements.txt`

```
fastapi==0.115.6
uvicorn[standard]==0.34.0
numpy>=1.26.4
scikit-learn>=1.5.0
pydantic>=2.10.0
xgboost>=2.0.0
```

| Package | Role |
|---|---|
| **fastapi** | Web framework for the prediction API |
| **uvicorn[standard]** | ASGI server to run FastAPI (`[standard]` includes `watchfiles` for auto-reload) |
| **numpy** | Array operations in `predictor.py` (`np.zeros`, feature vector construction) |
| **scikit-learn** | Required by the pickled model (XGBoost uses sklearn internally) |
| **pydantic** | Request/response validation schemas |
| **xgboost** | Required to deserialize the `.pkl` model (pickle needs the class definition) |

### `PROJECT_IMPLEMENTATION_GUIDE.md`

Interview preparation document with architecture explanation, prediction flow, threshold reasoning, and common interview questions.

---

## 8. System Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        TRAINING PHASE (Offline)                         │
│                                                                         │
│  ┌─────────────────────┐    ┌──────────────────────────────────────┐    │
│  │  E_Commerce_Dataset │    │       churn_prediction.ipynb         │    │
│  │      .xlsx          │───▶│                                      │    │
│  │  (5,630 rows ×      │    │  1. Load Excel (pandas)              │    │
│  │   20 columns)       │    │  2. Drop CustomerID                  │    │
│  └─────────────────────┘    │  3. EDA (seaborn, matplotlib)        │    │
│                              │  4. Impute missing values            │    │
│                              │     - SimpleImputer (mean)           │    │
│                              │     - IterativeImputer (RandomForest)│    │
│                              │  5. One-hot encode categoricals      │    │
│                              │     (pd.get_dummies)                 │    │
│                              │  6. Train/test split (80/20)         │    │
│                              │  7. SMOTE oversampling               │    │
│                              │  8. Train 8 models, compare          │    │
│                              │  9. Select XGBoost (best F1)         │    │
│                              │ 10. Drop low-importance features     │    │
│                              │ 11. Retrain final XGBoost            │    │
│                              │ 12. Export model + columns           │    │
│                              └──────────────┬───────────────────────┘    │
│                                              │                           │
│                              ┌───────────────▼───────────────────┐      │
│                              │    Model Artifacts                 │      │
│                              │  churn_prediction_model.pkl (285KB)│      │
│                              │  columns.json (18 features)        │      │
│                              └───────────────┬───────────────────┘      │
└──────────────────────────────────────────────┼──────────────────────────┘
                                               │
                    ┌──────────────────────────┘
                    │  loaded once at startup
                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       SERVING PHASE (Online)                            │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                    FastAPI Backend (:8000)                      │     │
│  │                                                                 │     │
│  │  ┌──────────┐    ┌──────────┐    ┌────────────────────────┐   │     │
│  │  │ main.py  │───▶│schemas.py│    │     predictor.py       │   │     │
│  │  │          │    │          │    │                         │   │     │
│  │  │ Routes:  │    │ Request: │    │ ChurnPredictor class:  │   │     │
│  │  │ GET      │    │ 15 fields│    │  __init__():           │   │     │
│  │  │ /health  │    │ + ranges │    │    load .pkl model     │   │     │
│  │  │          │    │          │    │    load columns.json   │   │     │
│  │  │ POST     │    │ Response:│    │                         │   │     │
│  │  │ /api/    │    │ predict  │    │  _prepare_input_array():│   │     │
│  │  │ predict  │───▶│ + prob   │    │    normalize strings   │   │     │
│  │  │          │    │ + thresh │    │    zero array (18)      │   │     │
│  │  │ CORS:    │    │          │    │    fill numerics        │   │     │
│  │  │ allow *  │    └──────────┘    │    one-hot encode       │   │     │
│  │  └──────────┘                    │                         │   │     │
│  │                                  │  predict():             │   │     │
│  │                                  │    predict_proba()      │   │     │
│  │                                  │    threshold > 0.4      │   │     │
│  │                                  │    → "Churn"/"Not Churn"│   │     │
│  │                                  └────────────────────────┘   │     │
│  └─────────────────────────────┬──────────────────────────────────┘     │
│                                │                                        │
│         JSON response          │  ▲  JSON request                       │
│         {prediction,           │  │  {tenure, citytier,                 │
│          churn_probability,    │  │   gender, ...}                      │
│          threshold}            │  │                                      │
│                                ▼  │                                      │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Frontend UI (:5500)                           │    │
│  │                                                                  │    │
│  │  ┌───────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐ │    │
│  │  │index.html │   │ app.js   │   │ styles.css │   │server.py │ │    │
│  │  │           │   │          │   │            │   │          │ │    │
│  │  │ Bootstrap │   │ Form     │   │ .churn     │   │ Python   │ │    │
│  │  │ 4 form    │   │ → JSON   │   │ = red      │   │ HTTP     │ │    │
│  │  │ 15 fields │   │ → fetch()│   │ .not-churn │   │ server   │ │    │
│  │  │ dropdowns │   │ → render │   │ = green    │   │ port 5500│ │    │
│  │  │ result div│   │ result   │   │ .hidden    │   │          │ │    │
│  │  └───────────┘   └──────────┘   └────────────┘   └──────────┘ │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                ▲                                        │
│                                │                                        │
│                         User Browser                                    │
└──────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

```
User fills form → app.js reads 15 fields → JSON.stringify() → fetch() POST
    → FastAPI receives JSON → Pydantic validates (422 if invalid)
    → predictor._prepare_input_array() normalizes + one-hot encodes → 18-element numpy array
    → model.predict_proba() → P(Churn) extracted
    → threshold check (> 0.4 = "Churn")
    → JSON response → app.js updates DOM → User sees result
```

---

## Setup & Run

### Prerequisites

- Python 3.13 (recommended)
- pip

### 1. Create Virtual Environment

```bash
cd customer-churn-prediction-application
py -3.13 -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Backend (Terminal 1)

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

**Available URLs:**

| URL | Purpose |
|---|---|
| `http://127.0.0.1:8000/docs` | Interactive Swagger API documentation |
| `http://127.0.0.1:8000/health` | Health check endpoint |
| `http://127.0.0.1:8000/api/predict` | Prediction endpoint (POST) |

### 4. Start Frontend (Terminal 2)

```bash
python frontend/server.py
```

**Open:** `http://127.0.0.1:5500`

---

## API Example

```bash
curl -X POST "http://127.0.0.1:8000/api/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"tenure\":10,\"citytier\":3,\"warehousetohome\":15,\"gender\":\"male\",\"hourspendonapp\":3,\"numberofdeviceregistered\":2,\"satisfactionscore\":4,\"maritalstatus\":\"single\",\"numberofaddress\":2,\"complain\":0,\"orderamounthikefromlastyear\":12,\"couponused\":1,\"ordercount\":5,\"daysincelastorder\":7,\"cashbackamount\":150.5}"
```

**Response:**

```json
{
  "prediction": "Churn",
  "churn_probability": 0.6123,
  "threshold": 0.4
}
```
