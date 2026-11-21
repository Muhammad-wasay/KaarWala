"""
KaarWala - shared feature building, validation and prediction.

Used by:
  - app/streamlit_app.py  (Streamlit front end)
  - app/main.py           (FastAPI backend + HTML front end)

Nothing in this file depends on Streamlit or FastAPI.
"""

import math
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "kaarwala_pipeline.joblib"
MERGED_PATH = ROOT / "data" / "kaarwala_merged.csv"
LOOKUP_PATH = ROOT / "data" / "market_lookup.csv"

CURRENT_YEAR = 2026
MIN_YEAR = 1961
MAX_MILEAGE = 1_000_000
MIN_ENGINE_CC = 50
MAX_ENGINE_CC = 8000
MIN_PRICE = 150_000
TEST_MAE = 459_818            # test-set MAE from notebook 02, Step 2.5
LOOKUP_CAP_YEAR = 2024        # same cap used when the market data was merged
LUXURY_THRESHOLD = 10_000_000

# Exact column order the saved pipeline was trained on
FEATURES = [
    "brand", "model", "variant", "year", "mileage_km", "fuel_type",
    "transmission", "engine_cc", "city", "market_supply_count",
    "market_avg_mileage", "vehicle_age", "km_per_year", "province",
    "mileage_vs_market",
]

# City -> province lists (copied from notebook 01)
SINDH_LIST = [
    "Karachi", "Mehrabpur", "Nawabshah", "Hyderabad", "Sukkur", "Shahdadpur",
    "Jacobabad", "Ghotki", "Mirpur Mathelo", "Moro", "Larkana", "Noudero",
    "Mirpur Khas", "Badin", "Khairpur Mir", "Tando Adam", "Dadu", "Daulatpur",
    "Pano Aqil", "Sanghar", "Daharki", "Makli", "Tando Muhammad Khan",
    "Thatta", "Shikarpur", "Umerkot", "Kotri", "Jamshoro", "Tando Allahyar",
    "Matiari", "Kandhkot", "Kashmore", "Tharparkar", "Mithi", "Sujawal", "Hala",
    "Sehwan",
]
KPK_LIST = [
    "Peshawar", "Mardan", "Abbottabad", "Mansehra", "Haripur", "Swabi",
    "Tarbela", "Kohat", "Buner", "Charsadda", "Topi", "Swat", "Nowshera",
    "Takhtbai", "Ghazi", "Dera Ismail Khan", "Jehangira", "Risalpur",
    "Swatmingora", "Mingora", "Bannu", "Nowshera Cantt", "Qalandarabad",
    "Akora Khattak", "Saidu Sharif", "Malakand Agency", "Chitral", "Dir",
    "Timergara", "Karak", "Hangu", "Tank", "Lakki Marwat", "Batkhela",
    "Shabqadar", "Parachinar", "Besham", "Balakot", "Naran",
]
BALOCHISTAN_LIST = [
    "Quetta", "Gwadar", "Hub", "Khuzdar", "Turbat", "Chaman", "Sibi", "Zhob",
    "Loralai", "Pishin", "Kalat", "Mastung", "Pasni", "Ormara",
    "Dera Murad Jamali", "Usta Mohammad", "Lasbela", "Ziarat", "Nushki",
]
AJK_LIST = [
    "Mirpur Ak", "Dadyal Ak", "Muzaffarabad", "Mangla", "Bhimber", "Kashmir",
    "Hattian", "Rawalakot", "Kotli", "Bagh", "Neelum", "Sudhanoti", "Haveli",
    "Gilgit", "Skardu", "Hunza",
]


def get_province(city: str) -> str:
    if city == "Islamabad":
        return "Islamabad"
    if city in SINDH_LIST:
        return "Sindh"
    if city in KPK_LIST:
        return "KPK"
    if city in BALOCHISTAN_LIST:
        return "Balochistan"
    if city in AJK_LIST:
        return "AJK"
    return "Punjab"


# ----------------------------------------------------------------------------
# Loading (cached once per process)
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_reference() -> pd.DataFrame:
    return pd.read_csv(MERGED_PATH)


@lru_cache(maxsize=1)
def load_lookup() -> pd.DataFrame:
    lookup = pd.read_csv(LOOKUP_PATH)
    lookup["brand_k"] = lookup["brand"].astype(str).str.strip().str.lower()
    lookup["model_k"] = lookup["model"].astype(str).str.strip().str.lower()
    return lookup


@lru_cache(maxsize=1)
def get_options() -> dict:
    """Dropdown values taken from the dataset, so spellings match training."""
    ref = load_reference().dropna(subset=["brand", "model"]).astype({"brand": str, "model": str})

    models_by_brand = {
        brand: sorted(group["model"].unique().tolist())
        for brand, group in ref.groupby("brand")
    }
    variants = {
        f"{brand}|{model}": sorted(group["variant"].dropna().astype(str).unique().tolist())
        for (brand, model), group in ref.groupby(["brand", "model"])
    }
    return {
        "brands": sorted(models_by_brand),
        "models_by_brand": models_by_brand,
        "variants": variants,
        "fuel_types": sorted(ref["fuel_type"].dropna().astype(str).unique().tolist()),
        "transmissions": sorted(ref["transmission"].dropna().astype(str).unique().tolist()),
        "cities": sorted(ref["city"].dropna().astype(str).unique().tolist()),
        "limits": {
            "min_year": MIN_YEAR,
            "max_year": CURRENT_YEAR,
            "max_mileage": MAX_MILEAGE,
            "min_engine_cc": MIN_ENGINE_CC,
            "max_engine_cc": MAX_ENGINE_CC,
        },
    }


# ----------------------------------------------------------------------------
# Input normalisation and validation
# ----------------------------------------------------------------------------
def _options_map(series: pd.Series) -> dict:
    """Map lowercase -> dataset spelling, e.g. 'toyota' -> 'Toyota'."""
    return {str(v).strip().lower(): str(v) for v in series.dropna().unique()}


def _canon(value, options: dict):
    if value is None:
        return None
    return options.get(str(value).strip().lower())


def _as_number(value, label, errors, integer=False):
    if value is None or str(value).strip() == "":
        errors.append(f"{label} is required.")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be a number.")
        return None
    if not math.isfinite(number):
        errors.append(f"{label} must be a finite number.")
        return None
    if integer:
        if not number.is_integer():
            errors.append(f"{label} must be a whole number.")
            return None
        return int(number)
    return number


def prepare_inputs(raw: dict):
    """
    Normalise the raw form/API input to dataset spellings and validate it.

    Returns (inputs, errors, warnings). If errors is non-empty, do not predict.
    """
    errors, warnings = [], []
    ref = load_reference()

    brand = _canon(raw.get("brand"), _options_map(ref["brand"]))
    if brand is None:
        errors.append("Please choose a valid brand.")

    model = None
    if brand:
        model = _canon(raw.get("model"), _options_map(ref.loc[ref["brand"] == brand, "model"]))
        if model is None:
            errors.append(f"'{raw.get('model')}' is not a known model of {brand}.")

    variant = None
    variant_raw = raw.get("variant")
    if variant_raw and brand and model:
        pool = ref.loc[(ref["brand"] == brand) & (ref["model"] == model), "variant"]
        variant = _canon(variant_raw, _options_map(pool))
        if variant is None:
            warnings.append(
                f"Variant '{variant_raw}' is not in the dataset, so it was treated as unknown."
            )

    year = _as_number(raw.get("year"), "Model year", errors, integer=True)
    if year is not None and not (MIN_YEAR <= year <= CURRENT_YEAR):
        errors.append(f"Model year must be between {MIN_YEAR} and {CURRENT_YEAR}.")

    mileage = _as_number(raw.get("mileage"), "Mileage", errors)
    if mileage is not None and not (0 <= mileage <= MAX_MILEAGE):
        errors.append(f"Mileage must be between 0 and {MAX_MILEAGE:,} km.")

    fuel = _canon(raw.get("fuel_type"), _options_map(ref["fuel_type"]))
    if fuel is None:
        errors.append("Please choose a valid fuel type.")

    transmission = _canon(raw.get("transmission"), _options_map(ref["transmission"]))
    if transmission is None:
        errors.append("Please choose a valid transmission.")

    city = _canon(raw.get("city"), _options_map(ref["city"]))
    if city is None:
        errors.append("Please choose a valid city.")

    if fuel == "Electric":
        engine_cc = 0.0  # electric cars have no engine; matches training
    else:
        engine_cc = _as_number(raw.get("engine_cc"), "Engine capacity", errors)
        if engine_cc is not None and not (MIN_ENGINE_CC <= engine_cc <= MAX_ENGINE_CC):
            errors.append(
                f"Engine capacity must be between {MIN_ENGINE_CC} and {MAX_ENGINE_CC:,} cc."
            )

    asking = _as_number(raw.get("asking_price") or 0, "Asking price", errors)
    if asking is not None and asking < 0:
        errors.append("Asking price cannot be negative.")

    inputs = {
        "brand": brand,
        "model": model,
        "variant": variant,
        "year": year,
        "mileage": mileage,
        "fuel_type": fuel,
        "transmission": transmission,
        "engine_cc": engine_cc,
        "city": city,
        "asking_price": asking or 0.0,
    }
    return inputs, errors, warnings


# ----------------------------------------------------------------------------
# Feature building and prediction
# ----------------------------------------------------------------------------
def market_stats(lookup: pd.DataFrame, brand: str, model: str, year: int):
    """Exact (brand, model, year) match first, then the model-level median."""
    b = brand.strip().lower()
    m = model.strip().lower()
    capped_year = min(year, LOOKUP_CAP_YEAR)

    rows = lookup[(lookup["brand_k"] == b) & (lookup["model_k"] == m)]
    rows = rows.dropna(subset=["market_avg_mileage"])
    if rows.empty:
        return None, None, "none"

    exact = rows[rows["year"] == capped_year]
    if not exact.empty:
        return (float(exact["market_supply_count"].iloc[0]),
                float(exact["market_avg_mileage"].iloc[0]),
                "exact")

    return (float(rows["market_supply_count"].median()),
            float(rows["market_avg_mileage"].median()),
            "model")


def build_features(inputs: dict):
    """Build the one-row DataFrame the saved pipeline expects."""
    year = inputs["year"]
    mileage = float(inputs["mileage"])
    vehicle_age = max(CURRENT_YEAR - year, 0)

    supply, avg, source = market_stats(load_lookup(), inputs["brand"], inputs["model"], year)
    if source == "none":
        ref = load_reference()
        supply = float(ref["market_supply_count"].median())
        avg = float(ref["market_avg_mileage"].median())

    row = {
        "brand": inputs["brand"],
        "model": inputs["model"],
        "variant": inputs["variant"] if inputs["variant"] else np.nan,
        "year": year,
        "mileage_km": mileage,
        "fuel_type": inputs["fuel_type"],
        "transmission": inputs["transmission"],
        "engine_cc": float(inputs["engine_cc"]),
        "city": inputs["city"],
        "market_supply_count": supply,
        "market_avg_mileage": avg,
        "vehicle_age": vehicle_age,
        "km_per_year": round(mileage / max(vehicle_age, 1)),
        "province": get_province(inputs["city"]),
        "mileage_vs_market": mileage / avg,
    }
    return pd.DataFrame([row], columns=FEATURES), source


def make_verdict(predicted: float, asking: float) -> dict:
    diff = asking - predicted
    pct = diff / predicted * 100
    if abs(diff) <= TEST_MAE:
        status = "fair"
    elif diff > 0:
        status = "above"
    else:
        status = "below"
    return {
        "status": status,
        "difference_pkr": round(diff),
        "difference_pct": round(pct, 1),
    }


def estimate(raw: dict) -> dict:
    """
    Full pipeline: normalise, validate, build features, predict.

    Check result["errors"] first. If it is empty, the other keys are filled in.
    """
    inputs, errors, warnings = prepare_inputs(raw)
    if errors:
        return {"errors": errors, "warnings": warnings}

    features, source = build_features(inputs)
    predicted = max(float(load_model().predict(features)[0]), MIN_PRICE)
    low = max(predicted - TEST_MAE, MIN_PRICE)
    high = predicted + TEST_MAE

    verdict = None
    if inputs["asking_price"] > 0:
        verdict = make_verdict(predicted, inputs["asking_price"])

    return {
        "errors": [],
        "warnings": warnings,
        "car": inputs,
        "predicted_pkr": round(predicted),
        "range_low_pkr": round(low),
        "range_high_pkr": round(high),
        "market_source": source,
        "high_value": predicted >= LUXURY_THRESHOLD,
        "verdict": verdict,
    }