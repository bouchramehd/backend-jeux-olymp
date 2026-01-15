from pathlib import Path

print("joblib exists:", Path("model/olympics_rf_pack.joblib").exists())
print("csv exists:", Path("model/country_medals_ml.csv").exists())
