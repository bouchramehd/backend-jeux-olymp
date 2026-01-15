import joblib
import sys
import sklearn
import numpy as np
import pandas as pd

print("python:", sys.version.split()[0])
print("sklearn:", sklearn.__version__)
print("numpy:", np.__version__)
print("pandas:", pd.__version__)

pack = joblib.load("model/olympics_rf_pack.joblib")
print("✅ Loaded model pack")
print("TYPE:", type(pack))

if isinstance(pack, dict):
    print("KEYS:", list(pack.keys()))
else:
    # pipeline / estimator
    print(pack)
