## 1. End-to-End Project Flow (Kaam Kaise Aage Barhega?)
Humara project 3 clear stages mein chalega. Har stage ka ek **Fixed Output File** hai jo agle member ka **Input** banega:
```text
[STAGE 1: Member 1]
Raw CSVs (10k + 59k) 
   └──> Notebook: notebooks/01_data_and_eda.ipynb
   └──> Output File: data/kaarwala_merged.csv (10,589 rows × 16 columns) + 5 EDA Plots
             │
             ▼
[STAGE 2: Member 2]
Reads: data/kaarwala_merged.csv
   └──> Notebook: notebooks/02_modeling_and_tuning.ipynb
   └──> Split (70/15/15) + Sklearn Pipeline + 4 Models + 5-Fold GridSearchCV
   └──> Output File: models/kaarwala_pipeline.joblib + Metrics Table (MAE, RMSE, R²)
             │
             ▼
[STAGE 3: Member 3]
Reads: models/kaarwala_pipeline.joblib
   └──> Folder: app/
   └──> 1. Streamlit Cloud App (app/streamlit_app.py)
   └──> 2. Hugging Face Hub & Live Space
   └──> 3. Local FastAPI Backend + HTML Frontend (app/main.py + app/index.html)
```
---
## 2. Locked Data Contract (15 Features + 1 Target = Total 16 Columns)
> **Rule for All Members:** Kisi bhi file mein in 16 columns ke spelling change nahi honge! Member 1 yehi columns save karega, Member 2 inhi par Pipeline banayega, aur Member 3 inhi par App form banayega.

### A. Categorical Features (Text — 7 Columns)
*(Member 2 in par `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")` lagayega)*
| # | Column Name | Type | Example Values | Description |
|---|---|---|---|---|
| 1 | `brand` | text | `"Toyota"`, `"Honda"`, `"Suzuki"` | Gari banane wali company |
| 2 | `model` | text | `"Corolla"`, `"Civic"`, `"Cultus"` | Gari ka model naam |
| 3 | `variant` | text | `"VXL"`, `"GLi"`, `"Oriel"` | Sub-model / trim |
| 4 | `fuel_type` | text | `"Petrol"`, `"Hybrid"`, `"Diesel"` | Fuel ki qisam |
| 5 | `transmission` | text | `"Automatic"`, `"Manual"` | Gear transmission |
| 6 | `city` | text | `"Lahore"`, `"Karachi"`, `"Islamabad"` | Jis shehar mein gari listed hai |
| 7 | `province` | text | `"Punjab"`, `"Sindh"`, `"Islamabad"`, `"KPK"` | Regional geographic grouping |

### B. Numerical Features (Numbers — 8 Columns)
*(Member 2 in par `SimpleImputer(strategy="median")` + `StandardScaler()` lagayega)*
| # | Column Name | Type | Source / Formula | Description |
|---|---|---|---|---|
| 8 | `year` | int | 10k File | Model year (1961 – 2026) |
| 9 | `mileage_km` | float | 10k File | Kitne kilometer chali hui hai |
| 10 | `engine_cc` | float | 10k File | Engine size in CC (EV ke liye 0.0) |
| 11 | `market_supply_count` | float | Merged from 59k File | 59k dataset mein is `(brand, model)` ki kul listings |
| 12 | `market_avg_mileage` | float | Merged from 59k File | 59k dataset mein is `(brand, model)` ki average mileage |
| 13 | `vehicle_age` | int | `2026 - year` | Gari kitne saal purani hai |
| 14 | `km_per_year` | float | `mileage_km / max(vehicle_age, 1)` | Har saal average kitne KM chali |
| 15 | `mileage_vs_market` | float | `mileage_km / market_avg_mileage` | Market average ke muqable mein kitni chali |

### C. Target Variable (Jawab — 1 Column)
| Column Name | Type | Description |
|---|---|---|
| **`price_in_pkr`** | int / float | Asking price in Pakistani Rupees (PKR) |
---
# 📚 KaarWala — Detailed Team Guide & ML Dictionary (For All 3 Members)

---

## 📖 Part 1: Machine Learning ke Mushkil Lafzon ke Asaan Matlab
*(Har member pehle ye 8 lafz parh le taake poore project aur Viva mein koi confusion na ho)*

1. **EDA (Exploratory Data Analysis):** Model banane se pehle data ke 5 Graphs bana kar dekhna ke data kaisa hai, kahan values khali hain, aur kahan ghalat numbers hain.
2. **Missing Values (NaN):** Table mein jahan koi khana khali reh gaya ho (maslan 5% gariyon ka `variant` nahi likha hua).
3. **Imputation (Khali Jagah Bharna):** Khali khano (`NaN`) ko delete karne ke bajaye unmein average/median number ya sab se aam naam bhar dena.
4. **Outliers (Ajeeb Numbers):** Aisi rows jinmein ghalti se namumkin number likha ho (jaise gari ki mileage 50 lakh km ya price 500 rupay). Inko hum training se pehle hata dete hain.
5. **Encoding (`OneHotEncoder`):** Machine Learning models ko English ke lafz (`"Toyota"`, `"Lahore"`, `"Petrol"`) samajh nahi aate, unhein sirf numbers (`0` aur `1`) samajh aate hain. Lafzon ko `0` aur `1` mein badalne ko **Encoding** kehte hain.
6. **Scaling (`StandardScaler`):** Ek column mein `year` `2020` hai aur doosre mein `mileage_km` `1,50,000` hai. Numbers ke size ko ek barabar paimane par laane ko **Scaling** kehte hain.
7. **Pipeline (`scikit-learn Pipeline`):** Ek aisi automated machine jismein Imputation + Encoding + Scaling + Model chhote dabbon ki tarah ek line mein judhe hote hain.
8. **Data Leakage (Paper Leak Hona):** Agar Test data (imtihan wale data) ki maloomat ghalti se Training ke waqt model ko mil jaye, to use Data Leakage kehte hain. Is se bachne ke liye pehle data split karte hain aur Pipeline sirf Training data par fit karte hain.

## 🛠️ Part 2: Har Member ke Kaam ki Poori Tafseel (Step-by-Step)

### 👤 MEMBER 1: Data Merge, Feature Engineering & EDA Lead
**Notebook:** `notebooks/01_data_and_eda.ipynb`

* **Step 1.1 (Load Data):**
  * **Kya karna hai:** `10k file` (`main`) aur `59k file` (`other`) ko `pd.read_csv()` se load karna hai.
  * **Kyun:** `10k file` mein 2026 ke taza rates, `city` aur `variant` hain; `59k file` (CC0 Public Domain) mein poore Pakistan ki 59,554 gariyon ka record hai.
* **Step 1.2 (Clean Keys & Merge):**
  * **Kya karna hai:** Dono files mein `brand` aur `model` ko chhote letters (`.str.lower()`) mein kar ke `59k file` se har gari ki **kul ginti (`market_supply_count`)** aur **average mileage (`market_avg_mileage`)** nikaal kar `10k file` ke aage `merge` karni hai.
  * **Kyun:** Miss ki shart hai *"combine at least two public sources"*.
* **Step 1.3 (6 Naye Features Banana — Feature Engineering):**
  * **Kya karna hai:** Existing columns se 6 naye columns banane hain:
    1. `vehicle_age` = `2026 - year` *(Gari kitne saal purani hai)*
    2. `km_per_year` = `mileage_km / vehicle_age` *(Har saal kitni chali)*
    3. `mileage_vs_market` = `mileage_km / market_avg_mileage` *(Market average se kam chali ya zyada)*
    4. `is_hybrid_or_ev` = Hybrid/Electric hai to `1` warna `0`
    5. `is_automatic` = Automatic hai to `1` warna `0`
    6. `is_big_city` = Karachi/Lahore/Islamabad/Rawalpindi hai to `1` warna `0`
  * **Kyun:** Miss ki shart hai **minimum 15 features** aur **feature engineering**. Is se hamare kul **15 features** ban jate hain.
* **Step 1.4 (Outlier Cleaning):**
  * **Kya karna hai:** Jin rows mein `mileage_flag` ya `engine_cc_flag` mein ghalti hai ya namumkin price/mileage hai, unko clean karna hai, aur `price_outlier_flag` ko drop karna hai (taake Target Leakage na ho).
* **Step 1.5 (5 EDA Graphs Banana):**
  * **Kya karna hai:** `matplotlib` aur `seaborn` se 5 plots banane hain:
    1. Price Distribution Histogram (Asli price vs Log price)
    2. Top 10 Car Brands Bar Chart
    3. Mileage vs Price Scatter Plot
    4. Model Year / Vehicle Age vs Price Boxplot
    5. Numerical Features Correlation Heatmap
* **Step 1.6 (Save Final Files for Member 2 & Member 3):**
  * **Kya karna hai:** Clean 18-column table ko `data/kaarwala_merged.csv` mein save karna hai, aur `market` table ko `data/market_lookup.csv` mein save karna hai.

---

### 👤 MEMBER 2: ML Pipeline, Model Training & Tuning Lead
**Notebook:** `notebooks/02_modeling_and_tuning.ipynb`

* **Step 2.1 (Train / Validation / Test Split — 70% / 15% / 15%):**
  * **Kya karna hai:** `kaarwala_merged.csv` ko 3 hisson mein baantna hai:
    * **70% Training Set:** Jis se models seekhenge.
    * **15% Validation Set:** Jis par hum 4 models ka muqabla karwayenge.
    * **15% Test Set:** Tijori mein band! Sirf sab se aakhir mein best model par **1 baar** chalega.
* **Step 2.2 (Scikit-Learn Preprocessing Pipeline):**
  * **Kya karna hai:** `ColumnTransformer` banana hai:
    * **6 Text Columns** (`brand, model, variant, fuel_type, transmission, city`) $\rightarrow$ `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`.
    * **11 Number Columns** $\rightarrow$ `SimpleImputer(strategy="median")` + `StandardScaler()`.
* **Step 2.3 (1 Baseline + 3 Models Train Karna):**
  * **Kya karna hai:** 4 models ko train kar ke Validation Set par check karna hai:
    1. **Baseline:** `LinearRegression` (Sab se saada model)
    2. **Model 1:** `KNeighborsRegressor` (KNN — Milti-julti gariyon ki price dekhne wala)
    3. **Model 2:** `RandomForestRegressor` (Bohot saare Decision Trees ka faisla)
    4. **Model 3:** `GradientBoostingRegressor` / `HistGradientBoostingRegressor` (Ghaltiyon se seekhne wala strong model)
* **Step 2.4 (Hyperparameter Tuning — `GridSearchCV`):**
  * **Kya karna hai:** Jo model sab se acha chale, us par **5-Fold Cross-Validation (`GridSearchCV`)** chala kar uski best settings (`n_estimators`, `max_depth`, `learning_rate`) dhoondni hain.
* **Step 2.5 (Final Evaluation Metrics — MAE, RMSE, R²):**
  * **Kya karna hai:** Held-out **Test Set (15%)** par final model chala kar 3 scores nikalne hain:
    * **$R^2$ Score:** Model kitne feesad (0 se 100%) sahi pattern pakad raha hai.
    * **MAE (Mean Absolute Error):** Model ki prediction mein average kitne Lakh rupay ka farq aata hai.
    * **RMSE (Root Mean Squared Error):** Bare errors par penalty dene wala metric.
    * Saath hi Train score vs Test score dekh kar batana hai ke **Overfitting** ya **Underfitting** to nahi hui.
* **Step 2.6 (Save Model):**
  * **Kya karna hai:** Final pipeline ko `joblib.dump()` se `models/kaarwala_pipeline.joblib` ke naam se save karna hai.

---

### 👤 MEMBER 3: 3-Way Deployment & Integration Lead
**Folder:** `app/`

* **Step 3.1 (How the App Works Behind the Scenes):**
  * User form mein sirf **9 asaan cheezein** bharega: *Brand, Model, Variant, Year, Mileage (KM), Engine (CC), Fuel Type, Transmission, City*.
  * App andar hi andar `market_lookup.csv` se *Market Supply* aur *Avg Mileage* uthayegi, aur 6 *Engineered Features* (`vehicle_age`, `km_per_year`, etc.) khud calculate kar ke **17 features** poore kar ke `kaarwala_pipeline.joblib` ko degi!
* **Step 3.2 (Deployment 1 — Streamlit Community Cloud):**
  * `app/streamlit_app.py` banana jismein:
    1. Car Input Form ho.
    2. Predicted Fair Price + Confidence Range (e.g., $\pm$ MAE range) dikhaye.
    3. Ek Explanatory Chart (maslan: usi model ki market price vs mileage chart) dikhaye.
    4. Isko **Streamlit Cloud** par host kar ke **Public URL** nikalna hai.
* **Step 3.3 (Deployment 2 — Hugging Face Hub & Space):**
  * Model pipeline aur Model Card (`README.md`) ko Hugging Face Hub par upload karna aur Gradio/Streamlit **Live Space** chalana.
* **Step 3.4 (Deployment 3 — Local FastAPI + HTML Web App):**
  * `app/main.py` (FastAPI backend jismein `/predict` endpoint ho aur ghalat input jaise negative mileage ya 1800 saal par **error message** de) + `app/index.html` (Frontend page) banana jo Demo ke waqt `localhost:8000` par chalega.