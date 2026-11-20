"""
KaarWala - FastAPI backend and HTML front end (local web app, deployment 3)

Run from the project root (E:\\KaarWala):
    uvicorn app.main:app --reload

Then open:
    http://localhost:8000        (HTML form)
    http://localhost:8000/docs   (API documentation, try /predict here)
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import sklearn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from features import (  # noqa: E402
    CURRENT_YEAR,
    MAX_ENGINE_CC,
    MAX_MILEAGE,
    MIN_ENGINE_CC,
    MIN_YEAR,
    estimate,
    get_options,
    load_model,
)

FIELD_LABELS = {
    "brand": "Brand",
    "model": "Model",
    "variant": "Variant",
    "year": "Model year",
    "mileage_km": "Mileage",
    "engine_cc": "Engine capacity",
    "fuel_type": "Fuel type",
    "transmission": "Transmission",
    "city": "City",
    "asking_price_pkr": "Asking price",
}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the model and dropdown data once at startup, so a broken file fails fast
    load_model()
    get_options()
    yield


app = FastAPI(
    title="KaarWala API",
    version="1.0.0",
    description="Fair market price estimate for used cars in Pakistan.",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")


class CarInput(BaseModel):
    brand: str = Field(..., min_length=1, max_length=60, examples=["Toyota"])
    model: str = Field(..., min_length=1, max_length=60, examples=["Corolla"])
    variant: Optional[str] = Field(default=None, max_length=120, examples=[None])
    year: int = Field(..., ge=MIN_YEAR, le=CURRENT_YEAR, examples=[2019])
    mileage_km: float = Field(..., ge=0, le=MAX_MILEAGE, examples=[60000])
    engine_cc: Optional[float] = Field(
        default=None, ge=0, le=MAX_ENGINE_CC,
        description="Required for petrol, diesel and hybrid. Ignored for Electric.",
        examples=[1600],
    )
    fuel_type: str = Field(..., min_length=1, examples=["Petrol"])
    transmission: str = Field(..., min_length=1, examples=["Automatic"])
    city: str = Field(..., min_length=1, examples=["Lahore"])
    asking_price_pkr: Optional[float] = Field(
        default=None, ge=0, description="Optional seller's asking price in PKR.",
    )


@app.exception_handler(RequestValidationError)
async def invalid_input(_request, exc: RequestValidationError):
    messages = []
    for err in exc.errors():
        loc = [str(p) for p in err.get("loc", []) if p != "body"]
        field = loc[-1] if loc else ""
        label = FIELD_LABELS.get(field, field or "Input")
        messages.append(f"{label}: {err.get('msg')}")
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid input.", "errors": messages, "warnings": []},
    )


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(APP_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_file": "kaarwala_pipeline.joblib",
        "scikit_learn": sklearn.__version__,
    }


@app.get("/options")
def options():
    """Dropdown values taken from the dataset."""
    return get_options()


@app.post("/predict")
def predict(car: CarInput):
    raw = {
        "brand": car.brand,
        "model": car.model,
        "variant": car.variant,
        "year": car.year,
        "mileage": car.mileage_km,
        "fuel_type": car.fuel_type,
        "transmission": car.transmission,
        "engine_cc": car.engine_cc,
        "city": car.city,
        "asking_price": car.asking_price_pkr,
    }
    result = estimate(raw)

    if result["errors"]:
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Invalid input.",
                "errors": result["errors"],
                "warnings": result["warnings"],
            },
        )

    return {
        "predicted_pkr": result["predicted_pkr"],
        "range_low_pkr": result["range_low_pkr"],
        "range_high_pkr": result["range_high_pkr"],
        "market_source": result["market_source"],
        "high_value": result["high_value"],
        "verdict": result["verdict"],
        "car": result["car"],
        "warnings": result["warnings"],
    }