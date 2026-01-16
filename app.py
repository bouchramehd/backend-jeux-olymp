from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
CSV_PATH = MODEL_DIR / "country_medals_ml.csv"
RF_GOLD_PATH = MODEL_DIR / "rf_gold.pkl"
RF_SILVER_PATH = MODEL_DIR / "rf_silver.pkl"
RF_BRONZE_PATH = MODEL_DIR / "rf_bronze.pkl"

# ----------------------------
# App
# ----------------------------
app = FastAPI(title="Olympics AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Load CSV
# ----------------------------
if not CSV_PATH.exists():
    raise RuntimeError(f"CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

required_cols = {"year", "country_name", "medal_type", "count", "past_medals", "avg_last_3", "delta"}
missing = required_cols - set(df.columns)
if missing:
    raise RuntimeError(f"CSV missing columns: {sorted(list(missing))}")

# normalize medal types to uppercase
df["medal_type"] = df["medal_type"].astype(str).str.upper().str.strip()
df["country_name"] = df["country_name"].astype(str).str.strip()
df["year"] = df["year"].astype(int)
df["count"] = df["count"].astype(float)

# ----------------------------
# Load Models Individuels
# ----------------------------

if not RF_GOLD_PATH.exists():
    raise RuntimeError(f"Model file not found: {RF_GOLD_PATH}")
if not RF_SILVER_PATH.exists():
    raise RuntimeError(f"Model file not found: {RF_SILVER_PATH}")
if not RF_BRONZE_PATH.exists():
    raise RuntimeError(f"Model file not found: {RF_BRONZE_PATH}")

# Charger les modèles
rf_gold = joblib.load(RF_GOLD_PATH)
rf_silver = joblib.load(RF_SILVER_PATH)
rf_bronze = joblib.load(RF_BRONZE_PATH)
# ----------------------------
# Helpers
# ----------------------------
def normalize_country(name: str) -> str:
    return str(name).strip()

def build_features(country_name: str, year: int) -> pd.DataFrame:
    """
    Build a single-row feature vector from historical data.
    If exact requested year doesn't exist in CSV, we compute based on closest past year available.
    """
    c = normalize_country(country_name)
    y = int(year)

    df_c = df[df["country_name"] == c].copy()
    if df_c.empty:
        raise HTTPException(status_code=404, detail="Country not found in dataset")

    # total medals per year (sum across medal types)
    by_year = (
        df_c.groupby("year")["count"]
        .sum()
        .reset_index()
        .sort_values("year")
    )

    # choose base year <= requested year, else fallback to max year
    past = by_year[by_year["year"] <= y]
    if past.empty:
        base_year = int(by_year["year"].max())
    else:
        base_year = int(past["year"].max())

    past_medals = float(by_year.loc[by_year["year"] == base_year, "count"].values[0])

    last3 = by_year[by_year["year"] <= base_year].tail(3)["count"].values
    avg_last_3 = float(np.mean(last3)) if len(last3) else 0.0

    prev = by_year[by_year["year"] < base_year].tail(1)["count"].values
    prev_val = float(prev[0]) if len(prev) else 0.0

    delta = float(past_medals - prev_val)

    # build row
    row = {
        "year": float(y),
        "past_medals": float(past_medals),
        "avg_last_3": float(avg_last_3),
        "delta": float(delta),
    }
    
    FEATURES = ["past_medals", "avg_last_3", "delta"]

    # ensure all features exist
    for f in FEATURES:
        if f not in row:
            row[f] = 0.0

    X = pd.DataFrame([row], columns=FEATURES).astype(float)
    return X

# ----------------------------
# Schemas
# ----------------------------
class PredictRequest(BaseModel):
    country_name: str
    year: int

# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Olympics AI API running"}

@app.get("/countries")
def get_countries():
    countries = sorted(df["country_name"].dropna().unique().tolist())
    return {"countries": countries}

@app.get("/years/{country_name}")
def get_years(country_name: str):
    c = normalize_country(country_name)
    df_c = df[df["country_name"] == c]
    if df_c.empty:
        raise HTTPException(status_code=404, detail="Country not found in dataset")
    years = sorted(df_c["year"].dropna().unique().astype(int).tolist())
    return {"country_name": c, "years": years}

@app.get("/medals/summary")
def medals_summary():
    """
    Return country summary:
    [{country,gold,silver,bronze,total}]
    """
    pivot = (
        df.pivot_table(
            index="country_name",
            columns="medal_type",
            values="count",
            aggfunc="sum",
            fill_value=0
        )
        .reset_index()
    )

    # ensure columns exist
    for m in ["GOLD", "SILVER", "BRONZE"]:
        if m not in pivot.columns:
            pivot[m] = 0

    pivot["total"] = pivot["GOLD"] + pivot["SILVER"] + pivot["BRONZE"]
    pivot = pivot.sort_values("total", ascending=False)

    out = []
    for _, r in pivot.iterrows():
        out.append({
            "country": r["country_name"],
            "gold": int(r["GOLD"]),
            "silver": int(r["SILVER"]),
            "bronze": int(r["BRONZE"]),
            "total": int(r["total"]),
        })
    return out

@app.get("/medals/history/{country_name}")
def medals_history(country_name: str):
    """
    Total medals per year for a given country:
    {country_name, years:[...], totals:[...]}
    """
    c = normalize_country(country_name)
    df_c = df[df["country_name"] == c]
    if df_c.empty:
        raise HTTPException(status_code=404, detail="Country not found in dataset")

    by_year = (
        df_c.groupby("year")["count"]
        .sum()
        .reset_index()
        .sort_values("year")
    )

    return {
        "country_name": c,
        "years": by_year["year"].astype(int).tolist(),
        "totals": by_year["count"].astype(int).tolist(),
    }

@app.post("/predict")
def predict(req: PredictRequest):
    c = normalize_country(req.country_name)
    y = int(req.year)

    X = build_features(c, y)

    g = float(rf_gold.predict(X)[0])
    s = float(rf_silver.predict(X)[0])
    b = float(rf_bronze.predict(X)[0])

    gold = max(0, int(round(g)))
    silver = max(0, int(round(s)))
    bronze = max(0, int(round(b)))

    return {
        "country_name": c,
        "year": y,
        "gold": gold,
        "silver": silver,
        "bronze": bronze,
        "total": gold + silver + bronze
    }