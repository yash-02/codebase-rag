import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df['Churn'] = df['Churn'].map({'Yes': 1, 'No':0})

# =====================
# Feature Engineering
# =====================

service_cols = [
    "PhoneService",
    "OnlineSecurity",
    "OnlineBackup",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

df["service_count"] = (
    df[service_cols]
    .eq("Yes")
    .sum(axis=1)
)

X = df.drop(
    columns=[
        "customerID",
        "Churn"
    ]
)

y = df["Churn"]

num_cols = X.select_dtypes(
    include=np.number
).columns.tolist()

cat_cols = X.select_dtypes(
    exclude=np.number
).columns.tolist()


preprocessor = ColumnTransformer([
    (
        "num",
        Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]),
        num_cols
    ),
    (
        "cat",
        Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]),
        cat_cols
    )
])

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=300,
            random_state=42
        )
    )
])

model.fit(X,y)

joblib.dump(
    model,
    MODELS_DIR / "churn_model.joblib"
)

# =====================
# Segmentation
# =====================

segment_cols = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "service_count"
]

seg_df = df[segment_cols].copy()

seg_df = seg_df.fillna(
    seg_df.median()
)

seg_scaler = StandardScaler()

seg_scaled = seg_scaler.fit_transform(
    seg_df
)

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

kmeans.fit(seg_scaled)

joblib.dump(
    seg_scaler,
    MODELS_DIR / "preprocessor.joblib"
)

joblib.dump(
    kmeans,
    MODELS_DIR / "kmeans.joblib"
)

print("Models saved successfully")
