"""
KaarWala - Fair Market Price Estimator (Streamlit app, Member 3)

Run from the project root:
    streamlit run app/streamlit_app.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

# ----------------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------------
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
TEST_MAE = 479_339
LUXURY_THRESHOLD = 10_000_000
LOOKUP_CAP_YEAR = 2024

FEATURES = [
    "brand", "model", "variant", "year", "mileage_km", "fuel_type",
    "transmission", "engine_cc", "city", "market_supply_count",
    "market_avg_mileage", "vehicle_age", "km_per_year", "province",
    "mileage_vs_market",
]

# Colours (light theme)
BG = "#f5f7fb"
CARD = "#ffffff"
BORDER = "#e2e8f0"
TEXT = "#0f172a"
MUTED = "#64748b"
ACCENT = "#f97316"
ACCENT2 = "#ec4899"
GOOD = "#16a34a"
BAD = "#dc2626"
INFO = "#0284c7"

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
# Loading (cached)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference() -> pd.DataFrame:
    return pd.read_csv(MERGED_PATH)


@st.cache_data
def load_lookup() -> pd.DataFrame:
    lookup = pd.read_csv(LOOKUP_PATH)
    lookup["brand_k"] = lookup["brand"].astype(str).str.strip().str.lower()
    lookup["model_k"] = lookup["model"].astype(str).str.strip().str.lower()
    return lookup


# ----------------------------------------------------------------------------
# Backend feature building
# ----------------------------------------------------------------------------
def market_stats(lookup: pd.DataFrame, brand: str, model: str, year: int):
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


def build_features(inp: dict, reference: pd.DataFrame, lookup: pd.DataFrame):
    year = inp["year"]
    mileage = float(inp["mileage"])
    vehicle_age = max(CURRENT_YEAR - year, 0)
    fuel = inp["fuel_type"]

    supply, avg, source = market_stats(lookup, inp["brand"], inp["model"], year)
    if source == "none":
        supply = float(reference["market_supply_count"].median())
        avg = float(reference["market_avg_mileage"].median())

    row = {
        "brand": inp["brand"],
        "model": inp["model"],
        "variant": inp["variant"] if inp["variant"] else np.nan,
        "year": year,
        "mileage_km": mileage,
        "fuel_type": fuel,
        "transmission": inp["transmission"],
        "engine_cc": 0.0 if fuel == "Electric" else float(inp["engine_cc"]),
        "city": inp["city"],
        "market_supply_count": supply,
        "market_avg_mileage": avg,
        "vehicle_age": vehicle_age,
        "km_per_year": round(mileage / max(vehicle_age, 1)),
        "province": get_province(inp["city"]),
        "mileage_vs_market": mileage / avg,
    }
    return pd.DataFrame([row], columns=FEATURES), source


def validate(inp: dict) -> list:
    errors = []
    if not inp["brand"] or not inp["model"]:
        errors.append("Please select a brand and model.")
    if not (MIN_YEAR <= inp["year"] <= CURRENT_YEAR):
        errors.append(f"Model year must be between {MIN_YEAR} and {CURRENT_YEAR}.")
    if not (0 <= inp["mileage"] <= MAX_MILEAGE):
        errors.append(f"Mileage must be between 0 and {MAX_MILEAGE:,} km.")
    if inp["fuel_type"] != "Electric" and not (MIN_ENGINE_CC <= inp["engine_cc"] <= MAX_ENGINE_CC):
        errors.append(f"Engine capacity must be between {MIN_ENGINE_CC} and {MAX_ENGINE_CC:,} cc.")
    if inp["asking_price"] < 0:
        errors.append("Asking price cannot be negative.")
    return errors


# ----------------------------------------------------------------------------
# Styling (light theme)
# ----------------------------------------------------------------------------
CSS = f"""
<style>
    html, body, [class*="css"] {{
        font-family: "Inter", "Segoe UI", system-ui, sans-serif;
    }}
    .stApp {{
        background: radial-gradient(1200px 600px at 10% -10%, #ffe8d6 0%, {BG} 55%);
    }}
    #MainMenu, footer {{ visibility: hidden; }}
    .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }}

    .hero {{
        background: linear-gradient(135deg, #ffffff 0%, #fff4ec 100%);
        border: 1px solid {BORDER};
        border-radius: 22px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.6rem;
        box-shadow: 0 16px 40px rgba(15,23,42,.08);
        position: relative;
        overflow: hidden;
    }}
    .hero::after {{
        content: "";
        position: absolute; right: -80px; top: -80px;
        width: 260px; height: 260px; border-radius: 50%;
        background: radial-gradient(circle, rgba(236,72,153,.18), transparent 70%);
    }}
    .hero h1 {{
        margin: 0; font-size: 2.8rem; font-weight: 800; letter-spacing: -0.5px;
        background: linear-gradient(90deg, {ACCENT}, {ACCENT2}, #7c3aed);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .hero p {{ margin: .5rem 0 0 0; color: {MUTED}; font-size: 1.05rem; }}
    .badges {{ margin-top: 1.1rem; display: flex; gap: .5rem; flex-wrap: wrap; }}
    .badge {{
        background: #ffffff; border: 1px solid {BORDER};
        color: {TEXT}; padding: .35rem .8rem; border-radius: 999px; font-size: .8rem;
        box-shadow: 0 2px 6px rgba(15,23,42,.05);
    }}

    .card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 20px;
        padding: 1.6rem 1.7rem;
        box-shadow: 0 10px 30px rgba(15,23,42,.06);
    }}
    .section-title {{
        display: flex; align-items: center; gap: .6rem;
        font-size: 1.1rem; font-weight: 700; color: {TEXT}; margin-bottom: .2rem;
    }}
    .section-sub {{ color: {MUTED}; font-size: .85rem; margin-bottom: 1rem; }}

    .chips {{ display: flex; flex-wrap: wrap; gap: .45rem; margin: .4rem 0 1rem 0; }}
    .chip {{
        background: #eff6ff; border: 1px solid #bfdbfe;
        color: {INFO}; padding: .25rem .65rem; border-radius: 8px; font-size: .78rem;
    }}

    .result-card {{
        background: linear-gradient(160deg, #ffffff 0%, #fff7f0 100%);
        border: 1px solid {BORDER};
        border-radius: 20px; padding: 1.8rem;
        box-shadow: 0 12px 32px rgba(15,23,42,.08);
    }}
    .label {{ color: {MUTED}; font-size: .8rem; text-transform: uppercase; letter-spacing: 1.2px; }}
    .big-price {{
        font-size: 3rem; font-weight: 800; line-height: 1.1; margin: .4rem 0 1.4rem 0;
        background: linear-gradient(90deg, {TEXT}, #334155);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .range-bar {{
        position: relative; height: 10px; border-radius: 999px; margin: .6rem 0 .4rem 0;
        background: linear-gradient(90deg, #cbd5e1, {ACCENT}, {ACCENT2});
    }}
    .range-dot {{
        position: absolute; top: 50%; width: 20px; height: 20px; border-radius: 50%;
        background: #ffffff; transform: translate(-50%, -50%);
        border: 3px solid {ACCENT};
        box-shadow: 0 4px 10px rgba(15,23,42,.2);
    }}
    .range-ends {{ display: flex; justify-content: space-between; color: {TEXT};
        font-size: .9rem; font-weight: 600; }}
    .src {{ color: {MUTED}; font-size: .8rem; margin-top: 1rem; }}

    .verdict {{
        border-radius: 14px; padding: 1rem 1.2rem; margin-top: 1rem;
        border: 1px solid; font-size: .92rem; color: {TEXT};
    }}
    .verdict .v-title {{ font-weight: 700; margin-bottom: .2rem; }}
    .verdict.ok   {{ border-color: #86efac; background: #f0fdf4; }}
    .verdict.bad  {{ border-color: #fca5a5; background: #fef2f2; }}
    .verdict.good {{ border-color: #93c5fd; background: #eff6ff; }}
    .warn {{
        border-radius: 14px; padding: .9rem 1.1rem; margin-top: .8rem;
        border: 1px solid #fcd34d; background: #fffbeb;
        color: #92400e; font-size: .88rem;
    }}

    .placeholder {{
        text-align: center; padding: 3.5rem 1.5rem; color: {MUTED};
    }}
    .placeholder .icon {{ font-size: 3.2rem; margin-bottom: .6rem; }}
    .placeholder h3 {{ color: {TEXT}; margin: .2rem 0 .4rem 0; font-weight: 700; }}

    div[data-testid="stVerticalBlockBorderWrapper"] > div {{
        background: {CARD};
        border-radius: 20px;
    }}
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div,
    div[data-testid="stNumberInput"] input {{
        border-radius: 10px !important;
    }}
    .stButton > button {{
        border-radius: 12px; height: 3rem; font-weight: 700; font-size: 1rem;
        background: linear-gradient(90deg, {ACCENT}, {ACCENT2}); border: none; color: white;
        box-shadow: 0 8px 20px rgba(236,72,153,.25);
    }}
    .stButton > button:hover {{ filter: brightness(1.05); color: white; border: none; }}
    label, .stMarkdown p {{ color: {TEXT}; }}
</style>
"""


def hero_html(n_listings: int) -> str:
    return f"""
<div class="hero">
    <h1>KaarWala</h1>
    <p>Know what a used car is really worth in Pakistan.</p>
    <div class="badges">
        <span class="badge">🚗 {n_listings:,} listings</span>
        <span class="badge">🌲 Random Forest model</span>
        <span class="badge">📊 15 features</span>
        <span class="badge">🇵🇰 Pakistan market</span>
    </div>
</div>
"""


def chips_html(items: list) -> str:
    inner = "".join(f'<span class="chip">{i}</span>' for i in items)
    return f'<div class="chips">{inner}</div>'


def result_html(predicted: float, low: float, high: float, source: str) -> str:
    span = high - low
    pos = 50.0 if span <= 0 else (predicted - low) / span * 100
    pos = min(max(pos, 0), 100)
    src_text = {
        "exact": "Market data: exact match for this model and year.",
        "model": "Market data: model-level average (no exact year match).",
        "none": "Market data: no listings for this model, overall averages used.",
    }[source]
    return f"""
<div class="result-card">
    <div class="label">Estimated fair price</div>
    <div class="big-price">{fmt_pkr(predicted)}</div>
    <div class="label">Likely range</div>
    <div class="range-bar"><div class="range-dot" style="left:{pos:.1f}%"></div></div>
    <div class="range-ends"><span>{fmt_pkr(low)}</span><span>{fmt_pkr(high)}</span></div>
    <div class="src">{src_text}</div>
</div>
"""


def verdict_html(predicted: float, asking: float) -> str:
    diff = asking - predicted
    pct = diff / predicted * 100
    if abs(diff) <= TEST_MAE:
        cls, title = "ok", "Looks fair"
        text = f"Asking price is within the typical error margin (±{fmt_pkr(TEST_MAE)}) of the estimate."
    elif diff > 0:
        cls, title = "bad", "Above market value"
        text = f"Asking is {fmt_pkr(diff)} ({pct:+.1f}%) above the estimate. Negotiate."
    else:
        cls, title = "good", "Below market value"
        text = f"Asking is {fmt_pkr(-diff)} ({pct:+.1f}%) below the estimate. Check the car's condition before buying."
    return f'<div class="verdict {cls}"><div class="v-title">{title}</div>{text}</div>'


def placeholder_html() -> str:
    return """
<div class="card placeholder">
    <div class="icon">🔍</div>
    <h3>Ready when you are</h3>
    <div>Fill in the car details on the left and press <b>Estimate fair price</b>.</div>
</div>
"""


def fmt_pkr(value: float) -> str:
    if value >= 10_000_000:
        return f"PKR {value / 10_000_000:.2f} Cr"
    return f"PKR {value / 100_000:.2f} L"


def price_chart(comps: pd.DataFrame, mileage: float, estimate: float, asking: float):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor(CARD)
    ax.set_facecolor(CARD)

    ax.scatter(comps["mileage_km"] / 1000, comps["price_in_pkr"] / 100_000,
               alpha=0.55, s=40, color=INFO, edgecolor="none", label="Dataset listings")
    ax.scatter([mileage / 1000], [estimate / 100_000], color=ACCENT, marker="*",
               s=320, zorder=3, edgecolor=TEXT, linewidth=0.6, label="Your car (estimate)")
    if asking:
        ax.scatter([mileage / 1000], [asking / 100_000], color=TEXT, marker="X",
                   s=140, zorder=3, label="Asking price")

    x_max = max(comps["mileage_km"].quantile(0.98) / 1000, mileage / 1000) * 1.1
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Mileage (thousand km)", color=MUTED)
    ax.set_ylabel("Price (Lakh PKR)", color=MUTED)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.grid(linestyle="--", alpha=0.5, color=BORDER)
    ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
st.set_page_config(page_title="KaarWala", page_icon="🚗", layout="wide",
                   initial_sidebar_state="collapsed")
st.html(CSS)

model = load_model()
reference = load_reference()
lookup = load_lookup()

st.html(hero_html(len(reference)))

left, right = st.columns([1, 1.2], gap="large")

with left:
    with st.container(border=True):
        st.html('<div class="section-title">🚘 Vehicle</div>'
                '<div class="section-sub">Tell us about the car you want to value.</div>')

        c1, c2 = st.columns(2)
        with c1:
            brand = st.selectbox("Brand", sorted(reference["brand"].dropna().unique()))
        with c2:
            models = sorted(reference.loc[reference["brand"] == brand, "model"].dropna().unique())
            model_name = st.selectbox("Model", models)

        variants = sorted(reference.loc[
            (reference["brand"] == brand) & (reference["model"] == model_name), "variant"
        ].dropna().unique())
        variant_choice = st.selectbox("Variant", ["Not sure / not listed"] + list(variants))
        variant = None if variant_choice == "Not sure / not listed" else variant_choice

        c3, c4 = st.columns(2)
        with c3:
            year = st.number_input("Model year", min_value=MIN_YEAR, max_value=CURRENT_YEAR,
                                   value=2018, step=1)
        with c4:
            mileage = st.number_input("Mileage (km)", min_value=0, max_value=MAX_MILEAGE,
                                      value=60000, step=1000)

    with st.container(border=True):
        st.html('<div class="section-title">⚙️ Specs</div>'
                '<div class="section-sub">Engine, fuel and gearbox.</div>')

        c5, c6 = st.columns(2)
        with c5:
            fuel_options = sorted(reference["fuel_type"].dropna().unique())
            fuel_type = st.selectbox("Fuel type", fuel_options,
                                     index=fuel_options.index("Petrol") if "Petrol" in fuel_options else 0)
        with c6:
            transmission = st.selectbox("Transmission",
                                        sorted(reference["transmission"].dropna().unique()))

        if fuel_type == "Electric":
            st.info("Electric cars have no engine, so 0 cc is used.")
            engine_cc = 0.0
        else:
            engine_cc = st.number_input("Engine (cc)", min_value=0, max_value=MAX_ENGINE_CC,
                                        value=1300, step=100)

    with st.container(border=True):
        st.html('<div class="section-title">📍 Listing</div>'
                '<div class="section-sub">Where it is sold and the seller\'s price (optional).</div>')

        city = st.selectbox("City", sorted(reference["city"].dropna().unique()),
                            index=None, placeholder="Choose a city")
        asking_price = st.number_input("Seller's asking price (PKR, optional)",
                                       min_value=0, value=0, step=50000,
                                       help="Leave at 0 to skip the price check.")

    estimate_clicked = st.button("✨ Estimate fair price", type="primary", use_container_width=True)

with right:
    if not estimate_clicked:
        st.html(placeholder_html())
    elif city is None:
        st.error("Please choose a city.")
    else:
        inputs = {
            "brand": brand, "model": model_name, "variant": variant, "year": int(year),
            "mileage": mileage, "fuel_type": fuel_type, "transmission": transmission,
            "engine_cc": engine_cc, "city": city, "asking_price": asking_price,
        }
        errors = validate(inputs)

        if errors:
            for err in errors:
                st.error(err)
        else:
            features, source = build_features(inputs, reference, lookup)
            predicted = max(float(model.predict(features)[0]), MIN_PRICE)
            low = max(predicted - TEST_MAE, MIN_PRICE)
            high = predicted + TEST_MAE

            st.html(chips_html([
                f"{brand} {model_name}",
                f"{int(year)}",
                f"{mileage:,.0f} km",
                fuel_type,
                transmission,
                city,
            ]))

            st.html(result_html(predicted, low, high, source))

            if asking_price > 0:
                st.html(verdict_html(predicted, asking_price))

            if predicted >= LUXURY_THRESHOLD:
                st.html('<div class="warn">⚠️ High-value car. The real error is usually much larger '
                        'than the range shown, so treat this as a rough guide.</div>')

            comps = reference[(reference["brand"] == brand) & (reference["model"] == model_name)]
            comps = comps.dropna(subset=["mileage_km", "price_in_pkr"])
            if not comps.empty:
                with st.container(border=True):
                    st.html('<div class="section-title">📈 Market comparison</div>'
                            f'<div class="section-sub">{len(comps)} listings of this model '
                            'in the dataset.</div>')
                    fig = price_chart(comps, mileage, predicted, asking_price)
                    st.pyplot(fig)
                    plt.close(fig)

st.write("")
with st.expander("How this estimate works"):
    st.markdown(
        "- You enter the car's details. The app adds vehicle age, km per year, province, and "
        "market supply and average mileage for the model.\n"
        "- The saved scikit-learn pipeline predicts the price.\n"
        f"- The range is the estimate ± the test-set error (about {fmt_pkr(TEST_MAE)}).\n"
        "- The chart uses listings from the dataset, not live OLX ads.\n"
        "- This is an estimate, not a valuation or financial advice.\n\n"
        f"*Model: {type(model).__name__} · scikit-learn {sklearn.__version__}*"
    )