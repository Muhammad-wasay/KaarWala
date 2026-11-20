# 🚗 KaarWala: Fair Market Price Prediction for Used Vehicles in Pakistan

**Course:** CT-378 Machine Learning (Fall 2026)  
**Team Name:** Mind Masters (`CT-24236`, `CT-24238`,`CT-24075`)  
**Task Type:** Supervised Learning — Regression  

---

## 📌 Overview
**KaarWala** is an end-to-end Machine Learning system designed to estimate fair market asking prices (in PKR) for used vehicles across Pakistan. By combining granular listing specifications with macro-market supply and mileage benchmarks, KaarWala helps everyday buyers and sellers navigate vehicle valuation transparently.

---

## 📊 Dataset & Data Integration
To satisfy the multi-source requirement (minimum 5,000 rows and 15 features) while preventing data leakage, this project integrates two public datasets:

| Dataset | Author / Source | Records | Coverage / Date | License | Role in Pipeline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pakistan Used Cars Dataset: Prices & Specs 2026** | Danish Javaid Meo (Kaggle) | 10,589 | Sept 28, 2026 | Educational & Research Use | Primary dataset with city, variant, specs, and 2026 asking prices |
| **Pakistan Used Car Prices — Car Price Prediction** | Muhammad Humayun Babar (Kaggle) | 59,554 | Sept 2026 | **CC0: Public Domain** | External reference corpus used to join `market_supply_count` and `market_avg_mileage` |

* **Privacy & Ethics:** Neither dataset contains Personally Identifiable Information (no seller names, phone numbers, or residential addresses).
* **Final Feature Matrix:** **10,589 rows × 15 features** + `price_pkr` (Target).

---

## 🛠️ Feature Set (15 Features + 1 Target)
1. **Original Features (9):** `brand`, `model`, `variant`, `year`, `mileage_km`, `fuel_type`, `transmission`, `engine_cc`, `city`
2. **Merged Market Features (2):** `market_supply_count`, `market_avg_mileage`
3. **Engineered Features (6):** `vehicle_age`, `km_per_year`, `mileage_vs_market`, `is_hybrid_or_ev`, `is_automatic`, `is_big_city`

---

## 📁 Repository Structure
```text
KaarWala/
├── data/                   # Raw and merged datasets
├── notebooks/              # Step-by-step EDA and model training notebooks
├── models/                 # Saved scikit-learn pipeline (.joblib)
├── app/                    # Streamlit, Hugging Face Space, and FastAPI apps
├── requirements.txt        # Pinned Python dependencies
└── README.md               # Project documentation


---

## 👤 STAGE 2: Machine Learning Pipeline, Model Benchmarking & Tuning (Member 2 — Completed ✅)

**Execution Notebook:** `notebooks/02_modeling_and_tuning.ipynb`  
**Production Artifact:** `models/kaarwala_pipeline.joblib`

---

### 1. ⚙️ Technical Challenges & Solutions

During model development, four engineering and data challenges were identified and methodically resolved:

* **Challenge 1: Electric Vehicle (EV) Engine Displacement Anomaly**  
  *Issue:* 92 Electric Cars (e.g., Porsche Taycan, BMW i5, BYD Atto) had missing `engine_cc` values (`NaN`) because EVs lack piston engines. Blind median imputation would have incorrectly assigned them a 1,300cc/1,500cc combustion engine profile.  
  *Solution:* We enforced a domain-level condition prior to splitting:  
  $$\text{if } \text{fuel\_type} == \text{'Electric'} \implies \text{engine\_cc} = 0.0$$  
  This allowed the one-hot encoded `fuel_type_Electric` and a `0.0` displacement to accurately signal a pure battery-electric drivetrain.

* **Challenge 2: Target Price Right-Skewness & Variance Instability**  
  *Issue:* Secondary market prices range from PKR 5.0 Lakh (Mehran) up to PKR 4.25 Crore (Audi e-tron). Raw scale regression struggled with heavy outlier variance, capping validation $R^2$ at ~90%.  
  *Solution:* Integrated Scikit-Learn’s `TransformedTargetRegressor` applying a **Log Transformation (`np.log1p`)** on price during training, and inverse exponent (`np.expm1`) on inference. This boosted validation $R^2$ from **90.46% to 94.44%** while reducing Mean Absolute Error.

* **Challenge 3: Multi-processing Thread Deadlock under Windows**  
  *Issue:* Running nested 5-Fold GridSearchCV across multi-tree estimators with `n_jobs=-1` caused Windows subprocess locks, freezing execution for 6+ minutes.  
  *Solution:* Switched to an optimized, parallel-safe KFold cross-validation routine, completing in **30 seconds**.

* **Challenge 4: Ultra-Luxury Segment Variance**  
  *Issue:* Scatter diagnostics confirmed high precision between PKR 0 to 1 Crore, with wider variance appearing only above PKR 2 Crore due to scarce data (<1% of listings).  
  *Solution:* We avoided over-parameterization to preserve robust real-world generalization across the mainstream market.

---

### 2. 📊 Implementation & Benchmark Results

1. **Step 2.1 — Three-Way Split (70% / 15% / 15%):**  
   10,589 listings split with `random_state=42` to eliminate data leakage:  
   *Training (70%):* 7,412 rows | *Validation (15%):* 1,588 rows | *Held-Out Test (15%):* 1,589 rows.

2. **Step 2.2 — Preprocessing Pipeline (`ColumnTransformer`):**  
   - **7 Categorical Features:** `['brand', 'model', 'variant', 'fuel_type', 'transmission', 'city', 'province']` $\rightarrow$ Most-Frequent Imputer + OneHotEncoder (`handle_unknown='ignore'`).
   - **8 Numerical Features:** `['year', 'mileage_km', 'engine_cc', 'market_supply_count', 'market_avg_mileage', 'vehicle_age', 'km_per_year', 'mileage_vs_market']` $\rightarrow$ Median Imputer + StandardScaler.

3. **Step 2.3 — Four-Model Benchmark (Validation Set):**  
   - Linear Regression (Baseline): $R^2 = 0.8831$ | $\text{MAE} = \text{PKR } 760,887$
   - K-Nearest Neighbors (KNN): $R^2 = 0.9259$ | $\text{MAE} = \text{PKR } 630,136$
   - HistGradientBoosting: $R^2 = 0.9151$ | $\text{MAE} = \text{PKR } 651,656$
   - **Random Forest (Champion 🏆):** $R^2 = 0.9526$ | $\text{MAE} = \text{PKR } 469,337$

4. **Step 2.4 — 5-Fold Cross Validation (V1 to V5):**  
   Fold scores ranged from 87.92% to 93.97%, achieving a **Mean $R^2$ of 90.46% (±0.0221)**, verifying model consistency.

5. **Step 2.5 — Final Held-Out Test Evaluation (15% Unseen Data):**  
   - **Test Set $R^2$ Score:** **91.55%**  
   - **Test Mean Absolute Error (MAE):** **PKR 479,339 (~4.79 Lakh PKR)**  
   - **Train vs. Test Gap:** **5.26%** (Well within the 10% threshold $\implies$ Zero Overfitting).

6. **Step 2.6 — Production Export:**  
   Exported complete pipeline to `models/kaarwala_pipeline.joblib`.

---

### 3. 🚀 Blueprint & Flow for Member 3 (UI & Deployment)

Member 3 can load the complete pipeline via `joblib.load('models/kaarwala_pipeline.joblib')` for the Streamlit web app:

1. **User Form (Collect 9 Inputs):**  
   *Brand, Model, Variant, Year, Mileage, Engine CC, Fuel Type, Transmission, City.*
2. **Automated Backend Feature Completion:**  
   - Derive `province` from city mapping.
   - Calculate `vehicle_age = 2026 - year` and `km_per_year = mileage_km / max(vehicle_age, 1)`.
   - Query `data/market_lookup.csv` by `(brand, model)` for `market_supply_count` and `market_avg_mileage`.
   - Calculate `mileage_vs_market = mileage_km / market_avg_mileage`.
   - Enforce `if fuel_type == 'Electric' \implies engine_cc = 0.0`.
3. **Inference & UI Display:**  
   Pass features into `model.predict(input_df)` and display:
   - **Estimated Fair Price** (e.g., *PKR 34.50 Lakh*).
   - **Confidence Window:** Fair market band ($\pm \text{MAE}$ of PKR ~4.8 Lakh).