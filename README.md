# MLflow Lab — Experiment Tracking with California Housing Dataset

## Overview
This lab demonstrates how to use **MLflow** to track machine learning experiments.
We train 4 different regression models on the California Housing dataset and compare
their performance using the MLflow UI.

## Dataset
- **Name:** California Housing Dataset
- **Source:** Built into `scikit-learn` (no download needed)
- **Size:** 20,640 samples, 8 features
- **Task:** Predict median house value (in $100,000s)
- **Features:** MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

## Models Trained
| Model | Key Parameters |
|---|---|
| Linear Regression | — |
| Ridge Regression | alpha=1.0 |
| Random Forest | n_estimators=50, max_depth=10 |
| Gradient Boosting | n_estimators=100, learning_rate=0.1 |

## Metrics Tracked
- **RMSE** — Root Mean Squared Error (lower is better)
- **MAE** — Mean Absolute Error (lower is better)
- **R2** — R-Squared score (higher is better, max=1.0)

## Project Structure
```
ML_Flow_LAB-1/
├── mlflow_lab.py       ← Main training script
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
├── mlflow.db           ← Auto-created by MLflow (SQLite)
├── mlruns/             ← Auto-created by MLflow (artifacts)
└── venv/               ← Virtual environment
```

## Setup & Run

### 1. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the training script
```bash
python mlflow_lab.py
```

### 4. Launch the MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### 5. Open in browser
```
http://127.0.0.1:5000
```
> If port 5000 is blocked, use `--port 5001` and open `http://127.0.0.1:5001`

## What to Explore in the UI
- Click **"california-housing-prediction"** experiment on the left
- Select multiple runs → click **Compare** to see side-by-side metrics
- Sort by **R2** or **RMSE** to find the best model
- Click any run to see its parameters, metrics, and saved model artifacts