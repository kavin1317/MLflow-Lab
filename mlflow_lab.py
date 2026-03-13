import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# 1. SETUP MLFLOW
# ─────────────────────────────────────────────
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("california-housing-prediction")

# ─────────────────────────────────────────────
# 2. LOAD DATASET (built-in, no download needed)
# Predict median house value from 8 features
# ─────────────────────────────────────────────
print("Loading California Housing dataset...")
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

print(f"Dataset shape: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"Target: Median house value (in $100,000s)")
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 3. HELPER FUNCTION
# ─────────────────────────────────────────────
def train_and_log(model, model_name, params, X_tr, X_te, y_tr, y_te):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        mae  = mean_absolute_error(y_te, y_pred)
        r2   = r2_score(y_te, y_pred)

        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae",  mae)
        mlflow.log_metric("r2",   r2)

        mlflow.set_tag("dataset", "california_housing")
        mlflow.set_tag("model_type", model_name)

        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"\n✅ Run: {model_name}")
        print(f"   RMSE : {rmse:.4f}")
        print(f"   MAE  : {mae:.4f}")
        print(f"   R2   : {r2:.4f}")

# ─────────────────────────────────────────────
# 4. RUN EXPERIMENTS
# ─────────────────────────────────────────────
train_and_log(
    LinearRegression(),
    "LinearRegression",
    {"model": "LinearRegression"},
    X_train_scaled, X_test_scaled, y_train, y_test
)

train_and_log(
    Ridge(alpha=1.0),
    "Ridge_alpha1",
    {"model": "Ridge", "alpha": 1.0},
    X_train_scaled, X_test_scaled, y_train, y_test
)

train_and_log(
    RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
    "RandomForest_n50_depth10",
    {"model": "RandomForest", "n_estimators": 50, "max_depth": 10},
    X_train, X_test, y_train, y_test
)

train_and_log(
    GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "GradientBoosting_lr0.1",
    {"model": "GradientBoosting", "n_estimators": 100, "learning_rate": 0.1},
    X_train, X_test, y_train, y_test
)

print("\n🎉 All runs complete!")
print("   Run: mlflow ui --backend-store-uri sqlite:///mlflow.db")
print("   Open: http://127.0.0.1:5000")