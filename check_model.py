import joblib
import sys
import sklearn
import numpy as np
import pandas as pd
from pathlib import Path

print("python:", sys.version.split()[0])
print("sklearn:", sklearn.__version__)
print("numpy:", np.__version__)
print("pandas:", pd.__version__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

RF_GOLD_PATH = MODEL_DIR / "rf_gold.pkl"
RF_SILVER_PATH = MODEL_DIR / "rf_silver.pkl"
RF_BRONZE_PATH = MODEL_DIR / "rf_bronze.pkl"

# Charger les modèles
rf_gold = joblib.load(RF_GOLD_PATH)
print("✅ Loaded rf_gold:", type(rf_gold))

rf_silver = joblib.load(RF_SILVER_PATH)
print("✅ Loaded rf_silver:", type(rf_silver))

rf_bronze = joblib.load(RF_BRONZE_PATH)
print("✅ Loaded rf_bronze:", type(rf_bronze))
