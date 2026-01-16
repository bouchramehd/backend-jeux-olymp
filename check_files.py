from pathlib import Path

print("csv exists:", Path("model/country_medals_ml.csv").exists())
print("rf_gold exists:", Path("model/rf_gold.pkl").exists())
print("rf_silver exists:", Path("model/rf_silver.pkl").exists())
print("rf_bronze exists:", Path("model/rf_bronze.pkl").exists())
