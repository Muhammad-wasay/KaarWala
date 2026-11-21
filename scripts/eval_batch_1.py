import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

ROOT = Path("C:/Users/M Wasay/VS CODE ALL LANG FOLDERS/5th_SEMESTER_PROJECTS/ML-PROJECT/KaarWala")
sys.path.insert(0, str(ROOT / "app"))

from features import FEATURES, load_lookup, load_reference, market_stats, get_province, CURRENT_YEAR

model = joblib.load(ROOT / "models" / "kaarwala_pipeline.joblib")
lookup = load_lookup()
ref = load_reference()

cars = [
    {"url": "https://www.olx.com.pk/item/mehran-6-modal-03041128664-iid-1115315481", "brand": "Suzuki", "model": "Mehran", "variant": "VX", "year": 2006, "mileage": 90000, "fuel": "Petrol", "trans": "Manual", "cc": 800, "city": "Gujrat", "price": 700000},
    {"url": "https://www.olx.com.pk/item/daihatsu-mira-e-s-l-sa-3-2023-iid-1118904716", "brand": "Daihatsu", "model": "Mira", "variant": "ES", "year": 2023, "mileage": 44000, "fuel": "Petrol", "trans": "Automatic", "cc": 660, "city": "Lahore", "price": 3890000},
    {"url": "https://www.olx.com.pk/item/haval-h6-2025-iid-1121668665", "brand": "Haval", "model": "H6", "variant": "1.5T", "year": 2025, "mileage": 30522, "fuel": "Petrol", "trans": "Automatic", "cc": 1500, "city": "Lahore", "price": 8500000},
    {"url": "https://www.olx.com.pk/item/haval-h6-2024-total-genuine-iid-1121287717", "brand": "Haval", "model": "H6", "variant": "HEV", "year": 2024, "mileage": 41000, "fuel": "Hybrid", "trans": "Automatic", "cc": 1500, "city": "Lahore", "price": 9250000},
    {"url": "https://www.olx.com.pk/item/honda-civic-reborn-2010-iid-1121800088", "brand": "Honda", "model": "Civic", "variant": "VTi Oriel Prosmatec 1.8", "year": 2010, "mileage": 88000, "fuel": "Petrol", "trans": "Automatic", "cc": 1800, "city": "Lahore", "price": 3190000},
    {"url": "https://www.olx.com.pk/item/honda-civic-2013-iid-1121478544", "brand": "Honda", "model": "Civic", "variant": "VTi Prosmatec 1.8", "year": 2013, "mileage": 85168, "fuel": "Petrol", "trans": "Automatic", "cc": 1800, "city": "Lahore", "price": 3700000},
    {"url": "https://www.olx.com.pk/item/toyota-fortuner-sigma-2022-iid-1121804305", "brand": "Toyota", "model": "Fortuner", "variant": "2.8 Sigma 4", "year": 2022, "mileage": 35000, "fuel": "Diesel", "trans": "Automatic", "cc": 2800, "city": "Lahore", "price": 14800000},
    {"url": "https://www.olx.com.pk/item/toyota-vitz-safety-edition-3-2019-iid-1121797958", "brand": "Toyota", "model": "Vitz", "variant": "F 1.0", "year": 2019, "mileage": 86000, "fuel": "Petrol", "trans": "Automatic", "cc": 1000, "city": "Lahore", "price": 3990000},
    {"url": "https://www.olx.com.pk/item/honda-city-13-ivtec-2016-model-iid-1121325253", "brand": "Honda", "model": "City", "variant": "i-VTEC 1.3", "year": 2016, "mileage": 99000, "fuel": "Petrol", "trans": "Manual", "cc": 1300, "city": "Lahore", "price": 2899000},
    {"url": "https://www.olx.com.pk/item/toyota-corolla-altis-2025-iid-1121382650", "brand": "Toyota", "model": "Corolla", "variant": "Altis 1.6", "year": 2025, "mileage": 20000, "fuel": "Petrol", "trans": "Automatic", "cc": 1600, "city": "Lahore", "price": 6600000},
    {"url": "https://www.olx.com.pk/item/toyota-corolla-altis-2021-iid-1121808140", "brand": "Toyota", "model": "Corolla", "variant": "Altis 1.6", "year": 2021, "mileage": 104000, "fuel": "Petrol", "trans": "Automatic", "cc": 1600, "city": "Lahore", "price": 5450000},
    {"url": "https://www.olx.com.pk/item/daihatsu-mira-2023-iid-1119482859", "brand": "Daihatsu", "model": "Mira", "variant": "ES", "year": 2023, "mileage": 61000, "fuel": "Petrol", "trans": "Automatic", "cc": 660, "city": "Lahore", "price": 3990000}
]

rows = []
for c in cars:
    age = max(CURRENT_YEAR - c["year"], 0)
    km_yr = round(c["mileage"] / max(age, 1), 0)
    supply, avg_m, src = market_stats(lookup, c["brand"], c["model"], c["year"])
    if src == "none":
        supply = float(ref["market_supply_count"].median())
        avg_m = float(ref["market_avg_mileage"].median())
    m_ratio = round(c["mileage"] / avg_m, 2) if avg_m > 0 else 1.0
    prov = get_province(c["city"])
    
    feat = {
        "brand": c["brand"], "model": c["model"], "variant": c["variant"],
        "year": c["year"], "mileage_km": c["mileage"], "fuel_type": c["fuel"],
        "transmission": c["trans"], "engine_cc": c["cc"], "city": c["city"],
        "market_supply_count": supply, "market_avg_mileage": avg_m,
        "vehicle_age": age, "km_per_year": km_yr, "province": prov,
        "mileage_vs_market": m_ratio
    }
    pred = float(model.predict(pd.DataFrame([feat], columns=FEATURES))[0])
    diff_pct = ((pred - c["price"]) / c["price"]) * 100
    rows.append({
        "Car": f"{c['brand']} {c['model']} ({c['year']})",
        "OLX Asking Price": f"PKR {c['price']:,}",
        "KaarWala Prediction": f"PKR {round(pred):,}",
        "Difference": f"{diff_pct:+.1f}%",
        "Verdict": "Fair" if abs(diff_pct) <= 12 else ("Above Market" if diff_pct > 12 else "Below Market")
    })

df_res = pd.DataFrame(rows)
print("\n" + "="*85)
print("       AUTHENTIC LIVE OLX CARS EVALUATION RESULT")
print("="*85)
print(df_res.to_string(index=False))
print("="*85)
