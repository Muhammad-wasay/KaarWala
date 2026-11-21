# KaarWala: Member 3 Handoff (Deployment)

Owner: Muhammad Subhan Khan (SubhanKhanTrekker05)
Branch: `frontend-and-deployment`

## Links
- **Hugging Face model repo:** https://huggingface.co/SubhanKhanTrekker05/kaarwala-car-price
  - Files: `kaarwala_pipeline.joblib` (65 MB), `requirements.txt`, `README.md` (model card)
  - Model card committed as "Add model card" (commit `9ccb720`)
- **GitHub branch:** https://github.com/Muhammad-wasay/KaarWala/tree/frontend-and-deployment
- **Hugging Face Space:** not created yet
- **Streamlit Community Cloud:** not deployed yet (needs an admin of the GitHub repo, see below)
- **Local FastAPI app:** runs at `http://localhost:8000` (`uvicorn app.main:app --reload`)

## What I added
| Path | What it does |
|---|---|
| `app/features.py` | Shared logic for both apps: province lists, input normalisation, validation, feature building, prediction, fair-price verdict |
| `app/main.py` | FastAPI backend: `/predict`, `/options`, `/health`, `/docs`. Returns 422 with readable messages for bad input |
| `app/index.html` | HTML form and result page |
| `app/static/style.css` | Styling for the HTML page |
| `app/static/app.js` | Front-end script: loads dropdowns, calls `/predict`, renders result and errors |
| `app/streamlit_app.py` | Streamlit version of the same app. Uses `features.py` |
| `requirements.txt` | Pinned versions: streamlit 1.66.1, pandas 3.1.0, numpy 2.5.3, scikit-learn 1.9.1, joblib 1.6.0, matplotlib 3.11.2, fastapi, uvicorn |
| `.gitignore` | Excludes `.venv/`, `__pycache__/`, helper files, and secrets |

## Why it's built this way
- The guidelines require three deployments: Streamlit Cloud, Hugging Face, and a local FastAPI app with HTML.
- Both front ends import `features.py`, so validation and feature building cannot drift apart.
- scikit-learn is pinned to 1.9.1 because the saved model was created with that version.
- Member 1 and Member 2's notebooks, data, and `.py` files were not edited.

## How the model was checked
- The saved model has the 15 features in the order in `FEATURES`.
- Test predictions were checked against the CSV: Toyota Corolla 2019, 60,000 km, Lahore returned about PKR 53.2 lakh with range 48.4 to 58.0 lakh.
- Changing the year to 2015 lowers the price. Changing mileage to 150,000 km lowers it.
- Invalid years (18, 1800) return HTTP 422.

## Still open (team)
- **Model provenance (Member 2):** `joblib.dump()` is not in the notebook. Confirm which code created `kaarwala_pipeline.joblib`.
- **Metrics (Member 2):** the README's "90.46% → 94.44%" claim compares a CV mean with a validation score. The model card's test R² 0.9155 and MAE PKR 479,339 need confirming against the final model.
- **Lookup key (Member 1):** `market_lookup.csv` is per (brand, model, year). The docs say per (brand, model). The app matches the actual file.
- **Outlier cleaning (Member 1):** Step 1.4 is not in the code.
- **Feature count:** `PROJECT_DETAILS.md` says 17 features. The model uses 15.
- **Source of `data/olx_test_cars.csv`:** no URLs or scrape dates are recorded.
- **Streamlit deployment:** needs someone with admin rights on the GitHub repo, or a fork.

## How to check this in 5 minutes
1. Open the Hugging Face model link and the model card.
2. Open the GitHub branch link and go to `app/`.
3. Run locally:

`git checkout frontend-and-deployment`
`pip install -r requirements.txt`
`uvicorn app.main:app --reload`

Open `http://localhost:8000` and submit Toyota / Corolla / 2019 / 60000 / Petrol / Automatic / 1600 / Lahore.

## If you change the model or data
1. Edit `features.py` only for province lists, `FEATURES` order, `LOOKUP_CAP_YEAR`, and `TEST_MAE`.
2. Re-upload `kaarwala_pipeline.joblib` to the Hugging Face model repo.
3. Update the metrics on the model card.
4. Run the local app before pushing.