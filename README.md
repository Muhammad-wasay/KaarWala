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
* **Final Feature Matrix:** **10,589 rows × 17 features** + `price_pkr` (Target).

---

## 🛠️ Feature Set (17 Features)
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
