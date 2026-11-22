# KaarWala: Fair Market Price Prediction for Used Vehicles in Pakistan

**CT-378 Machine Learning — Semester Project Report | Fall 2026**
**Team: Mind Masters**
**Muhammad Wasay Rafiq (CT-24236) · Mesum Abbas (CT-24238) · Muhammad Subhan Khan (CT-24075)**

---

## Abstract

KaarWala is an end-to-end regression system that estimates the fair asking price of a used car in Pakistan from its specifications, usage history, and location. We combined two public Kaggle datasets sourced from OLX Pakistan and PakWheels: a primary dataset of 10,589 listings enriched with market statistics computed from a reference corpus of 59,554 listings, yielding 15 input features. A leakage-free scikit-learn Pipeline was used to compare a Linear Regression baseline against KNN, Random Forest, and HistGradientBoosting, with the best model tuned via manual hyperparameter search guided by 5-fold cross-validation. The final Random Forest model reached **R² = 94.36%, MAE = PKR 459,818, RMSE = PKR 1,325,946, and MAPE = 10.8%** on the held-out test set, surpassing all success targets. The pipeline is deployed via Streamlit Community Cloud, Hugging Face Spaces, and a local FastAPI web app.

---

## 1. Problem and Motivation

Buying or selling a used car is one of the largest financial decisions for Pakistani households, yet no objective reference price exists. High inflation, currency fluctuation, and regional registration differences make prices change rapidly — two nearly identical cars are often advertised at wildly different prices. Buyers risk overpaying; sellers either underprice their car or wait months at an unrealistic listing price.

**KaarWala** solves this by giving individual buyers and sellers a data-driven estimate of a fair asking price from the car's brand, model, year, mileage, engine, fuel type, transmission, and city — so both sides can negotiate with evidence instead of guesswork.

---

## 2. Data Collection

| Attribute | Dataset 1 — Primary | Dataset 2 — Market Reference |
|-----------|--------------------|-----------------------------|
| **Title / Author** | Pakistan Used Cars Dataset: Prices & Specs 2026 — Danish Javaid Meo | Pakistan Used Car Prices — Car Price Prediction — Muhammad Humayun Babar |
| **Source URL** | kaggle.com/datasets/danishjmeo/pakistan-used-cars-dataset-prices-and-specs-2026 | kaggle.com/datasets/muhammadhumyun/pakistan-used-car-prices-car-price-prediction |
| **Collection Method** | Scraped snapshot of publicly visible OLX Pakistan listings (by dataset author) | Cleaned archive of public PakWheels listings (by dataset author) |
| **Date** | Collected 28 September 2026 | Published/Updated 26–30 September 2026 |
| **Licence** | Research & educational use (stated in dataset description) | CC0: Public Domain |
| **Size** | 10,589 rows × 17 columns | 59,554 rows × 9 columns |
| **Personal Information** | None — no seller names, phone numbers, or photos | None — vehicle specifications and prices only |

**How the sources were combined.** The two datasets were NOT stacked row-wise — the reference corpus has no city or variant column, is older (model years up to 2024), and its prices are capped at PKR 10 million. Instead we joined them at the feature level: Dataset 2 was aggregated by normalised `(brand, model, year)` to compute `market_supply_count` (number of listings) and `market_avg_mileage` (average kilometres driven), and these were LEFT JOINed onto Dataset 1, keeping its 10,589 rows intact. Approximately 9% of primary rows found no match (mostly 2025–26 models); these values are imputed inside the Pipeline. Prices from the reference dataset were never used as features or targets.

**Final feature set (15 features + 1 target):**

| Group | Features |
|-------|---------|
| Original (9) | `brand`, `model`, `variant`, `year`, `mileage_km`, `fuel_type`, `transmission`, `engine_cc`, `city` |
| Merged Market (2) | `market_supply_count`, `market_avg_mileage` |
| Engineered (4) | `vehicle_age` (2026 − year), `km_per_year` (mileage ÷ age), `province` (city-to-province mapping), `mileage_vs_market` (mileage ÷ market_avg_mileage) |
| **Target (1)** | **`price_in_pkr`** — asking price in PKR |

---

## 3. Exploratory Data Analysis

Five plots were produced in `notebooks/01_eda.ipynb`:

| Figure | Plot | Key Finding |
|--------|------|------------|
| 1 | **Price distribution (histogram)** | Price is heavily right-skewed: median = PKR 3.43 lakh, mean = PKR 4.93 lakh, max = PKR 10.80 crore (luxury EVs). Log-transform normalises the distribution, justifying the use of `TransformedTargetRegressor(func=np.log1p)`. |
| 2 | **Top 10 car brands (bar chart)** | Toyota (3,110 listings, 29.4%), Suzuki (2,608, 24.6%), and Honda (2,196, 20.7%) together account for 74.7% of the market. |
| 3 | **Mileage vs. price (scatter plot)** | Strong negative correlation — higher mileage cars sell at lower prices. Outliers visible for recent low-mileage luxury imports. |
| 4 | **Vehicle age vs. price (box plot)** | Median price drops monotonically with age for most segments; newer (0–3 year old) cars show the widest price spread due to luxury imports. |
| 5 | **Correlation heatmap (numerical features)** | `year` shows strongest positive correlation with price (+0.61); `vehicle_age` shows strongest negative (−0.61); `mileage_km` moderately negative (−0.37). `market_supply_count` and `mileage_vs_market` add independent signal. |

**Missing values:** `variant` has the most gaps (548 rows, 5.2%). `market_supply_count` and `market_avg_mileage` are missing for 953 rows (9%) — cars whose `(brand, model, year)` did not appear in the reference corpus. All missing values are handled inside the Pipeline via median imputation (numerical) and most-frequent imputation (categorical), fitted only on training data.

---

## 4. Methodology and Design Decisions

**Data split:** 10,589 rows were split 70/15/15 with `random_state=42`:
- **Training (70%):** 7,412 rows — model training only
- **Validation (15%):** 1,588 rows — model comparison and selection
- **Test (15%):** 1,589 rows — evaluated once at the end, never touched during training

**Preprocessing Pipeline (`ColumnTransformer`, fitted on training data only):**
- 7 categorical features → `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown='ignore')`
- 8 numerical features → `SimpleImputer(median)` + `StandardScaler()`

**Key design decisions:**

| Decision | Reason |
|----------|--------|
| Feature-level join, not row stacking | Reference data has no city/variant, capped prices — stacking would corrupt the target |
| Join on `(brand, model, year)` | A 2005 and 2022 Corolla have very different typical mileage; year-level benchmark makes `mileage_vs_market` meaningful |
| `province` instead of city flags | City has 100+ values with sparse listings; province (5 values) captures regional price difference cleanly |
| `price_outlier_flag` dropped | Derived from price — using it as a feature would leak the target to the model |
| `TransformedTargetRegressor(np.log1p)` | Raw price is right-skewed; log-transform stabilises variance and improved validation R² from 90.46% to 95.26% |
| `engine_cc = 0.0` for EVs | 92 Electric cars had NaN engine displacement; median imputation would wrongly assign 1,300cc — domain fix applied before splitting |
| All preprocessing inside Pipeline | Prevents any data leakage; identical preprocessing is reused in all three deployed apps |

**Models compared (on validation set):**

| Model | Val R² | Val MAE (PKR) | Val RMSE (PKR) |
|-------|--------|--------------|----------------|
| Linear Regression (baseline) | 88.31% | 760,887 | — |
| KNN Regressor (k=5) | 92.59% | 630,136 | — |
| HistGradientBoosting | 91.51% | 651,656 | — |
| **Random Forest (Champion ✅)** | **95.26%** | **469,337** | — |

**Hyperparameter tuning:** The Random Forest champion was tuned by evaluating combinations of `n_estimators ∈ {100, 150}`, `max_depth ∈ {20, 25}`, and `min_samples_split ∈ {2, 4}` using 5-fold `KFold` cross-validation on the training set. Best configuration: `n_estimators=150, max_depth=25, min_samples_split=4`. Note: full `GridSearchCV` was attempted but caused Windows subprocess deadlocks with `n_jobs=-1`; iterative manual tuning on the validation set was used instead.

**5-Fold Cross-Validation results (R²):**
Fold V1: 87.92% | V2: 90.11% | V3: 91.44% | V4: 93.97% | V5: 88.86%
**Mean = 90.46% (±2.21%)** — confirms consistent performance across all data slices.

---

## 5. Results

**Success targets from proposal: R² ≥ 0.85 and MAPE ≤ 15% → Both met. ✅**

| Metric | Training Set | Test Set (held-out) |
|--------|-------------|-------------------|
| R² | 92.59% | **94.36%** |
| MAE | PKR 327,887 | **PKR 459,818 (~4.60 Lakh)** |
| RMSE | PKR 1,733,586 | **PKR 1,325,946 (~13.26 Lakh)** |
| MAPE | 5.7% | **10.8%** |

> **Why MAE and MAPE are most meaningful:** R² fluctuates when a few expensive luxury cars land in the test set. MAE tells users "on average, our estimate is off by ~4.6 lakh PKR"; MAPE = 10.8% means for a PKR 46 lakh car, the typical error is ~5 lakh.

**Data leakage verification:** A model trained on shuffled (randomised) price labels achieved test R² = −0.067 ≈ 0, confirming the model learned from genuine patterns, not any data leakage.

**Overfitting analysis:** Train MAPE = 5.7% vs Test MAPE = 10.8% — a moderate gap, expected because Random Forest memorises training data to some degree. The gap is within acceptable range; the leakage test and baseline comparison confirm the model is genuinely generalising.

**Where the model is least accurate:** Rare luxury cars (Porsche, Audi, BMW) priced above PKR 2 crore — fewer than 1% of training listings, so the model has limited data to learn their patterns. RMSE is higher than MAE because large errors on these cars are squared.

**External validation (OLX Pakistan, 73 verified listings with URLs):**
- **Test Dataset:** data/olx_verified_live_test.csv (contains 73 verified live listings with clickable URLs and unique Ad IDs across Pakistan)
- **External Evaluation:** ^2 = \mathbf{97.75\%}$ | **MAE:** PKR 309,416 (~3.09 Lakh) | **MAPE:** $\mathbf{7.90\%}$ (surpasses the $\le 15\%$ proposal target) ✅

*[Insert Figure: Predicted vs Actual price scatter on test set]*
*[Insert Figure: Random Forest feature importances]*

---

## 6. Deployment

To satisfy the guidelines requirement of three distinct deployment channels, KaarWala was packaged and deployed across cloud and local environments. All three interfaces load the single production pipeline artifact (`models/kaarwala_pipeline.joblib`) and share a centralized pre-inference validation and feature derivation module (`app/features.py`), ensuring zero prediction drift between platforms.

The user inputs 9 intuitive car specifications (Brand, Model, Variant, Year, Mileage, Fuel Type, Transmission, Engine CC, and City). The backend automatically derives the remaining 6 features (`province`, `vehicle_age`, `km_per_year`, `market_supply_count`, `market_avg_mileage`, and `mileage_vs_market`) before feeding them into the pipeline.

| Platform | Deployment Role & Architecture | Live Access / Endpoint |
| :--- | :--- | :--- |
| **Streamlit Community Cloud** (Deployment 1) | **Public Cloud Web App:** Hosted on Streamlit Community Cloud. Features a responsive valuation dashboard, dynamic dropdowns, confidence interval bar, fair/negotiate pricing verdict chips, and a real-time scatter comparison plot. | **Live URL:** [https://kaarwala.streamlit.app/](https://kaarwala.streamlit.app/) |
| **Hugging Face Model Hub & Space** (Deployment 2) | **Open-Source Model Registry & Space:** Production model card, dependencies (`requirements.txt`), and the serialized `kaarwala_pipeline.joblib` hosted on the Hugging Face Model Hub, paired with a live public inference Space. | **Model Hub:** [huggingface.co/SubhanKhanTrekker05/kaarwala-car-price](https://huggingface.co/SubhanKhanTrekker05/kaarwala-car-price)<br>**Live Space:** [https://subhankhantrekker05-kaarwala.static.hf.space/index.html](https://subhankhantrekker05-kaarwala.static.hf.space/index.html) |
| **Local FastAPI + HTML Web App** (Deployment 3) | **Production REST API & Local Web GUI:** High-performance FastAPI server running on Uvicorn (`http://localhost:8000`). Serves a lightweight vanilla HTML/CSS/JS frontend alongside a JSON `/predict` endpoint featuring strict Pydantic input validation (rejects invalid years like 1800 or negative mileage with descriptive HTTP 422 errors) and interactive Swagger documentation. | **Local Endpoint:** `http://localhost:8000`<br>**Interactive API Docs:** `http://localhost:8000/docs` |

### Architectural Safeguards in Deployment:
1. **Shared Feature Module (`app/features.py`):** Prevents discrepancy by keeping dictionary normalization (`toyota` $\rightarrow$ `Toyota`), province extraction, and market lookup consistent across both FastAPI and Streamlit.
2. **Pydantic Data Validation:** Ensures that malicious or erroneous inputs (e.g. mileage > 1,000,000 km, future model years, or missing cities) return clear, readable error toasts rather than crashing the regression estimator.
3. **Electric Vehicle Drivetrain Zeroing:** Automatically suppresses engine displacement to `0.0 cc` whenever `fuel_type == 'Electric'` is submitted, mirroring the exact condition enforced during model training.

---

## 7. Limitations and Future Work

**Limitations:**
- Prices are **asking prices**, not final transaction prices (buyers typically negotiate 5–15% below asking).
- No data on **vehicle condition**, accident history, or number of previous owners — features that strongly affect real prices.
- Some multi-word model names are split inconsistently in the source data (e.g., "Santa Fe" appears as model "Santa", variant "Fe Signature"), reducing join match rate.
- The reference corpus (Dataset 2) is older, has prices capped at PKR 10 million, and contains ~13% duplicate rows — only mileage and listing-count statistics were used from it.
- ~9% of rows had no market statistics match and were imputed; predictions for rare models may be less accurate.

**Future work:**
- Collect data regularly and re-train to account for inflation and PKR depreciation.
- Add vehicle condition score, listing images (image-based features), and accident history.
- Predict a **price range** (quantile regression) instead of a point estimate.
- Use `RandomizedSearchCV` or Optuna for more thorough hyperparameter search.
- Expand to cover motorcycles and commercial vehicles.

---

## 8. References

1. Meo, D. J. *Pakistan Used Cars Dataset: Prices & Specs 2026.* Kaggle, 2026.
   https://www.kaggle.com/datasets/danishjmeo/pakistan-used-cars-dataset-prices-and-specs-2026

2. Babar, M. H. *Pakistan Used Car Prices — Car Price Prediction.* Kaggle, 2026.
   https://www.kaggle.com/datasets/muhammadhumyun/pakistan-used-car-prices-car-price-prediction

3. OLX Pakistan. Original source of vehicle listings in Dataset 1.
   https://www.olx.com.pk

4. PakWheels. Original source of vehicle listings in Dataset 2.
   https://www.pakwheels.com

5. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python.
   *Journal of Machine Learning Research*, 12, 2825–2830.

6. Marvi, M. CT-378 Machine Learning lecture slides, Fall 2026.
   NED University of Engineering and Technology.

---

## 9. Individual Contribution

| Member | Roll No. | Contribution |
|--------|----------|-------------|
| **Muhammad Wasay Rafiq** | CT-24236 | Data sourcing, merging two Kaggle datasets, EDA (5 plots), feature engineering (province mapping, km_per_year, mileage_vs_market), outlier cleaning, final `kaarwala_merged.csv` and `market_lookup.csv` generation (`notebooks/01_eda.ipynb`) |
| **Mesum Abbas** | CT-24238 | Scikit-learn preprocessing Pipeline design, 4-model benchmark, 5-fold cross-validation, hyperparameter tuning, log-transform implementation, final test set evaluation, joblib model export (`notebooks/02_modeling_and_tuning.ipynb`, `models/kaarwala_pipeline.joblib`) |
| **Muhammad Subhan Khan** | CT-24075 | Streamlit web app, Hugging Face model card and Space, local FastAPI backend with `/predict` endpoint and HTML front end, input validation and error handling (`app/`) |

---

> [!NOTE]
> **Checklist Before Final PDF Export / Submission:**
> 1. Take a screenshot of the live Streamlit dashboard (`https://kaarwala.streamlit.app/`) and insert it in the report.
> 2. Take a screenshot of the FastAPI interactive Swagger UI (`http://localhost:8000/docs`) and insert it.
> 3. Verify that the Hugging Face Space has completed its build.
> 4. Export notebook plots (Predicted vs Actual, Feature Importance) into the results section.
