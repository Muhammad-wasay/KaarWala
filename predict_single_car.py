"""
KaarWala — Single Car Live Market Price Predictor (Terminal Tool)
Usage: python predict_single_car.py
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

def normalize_transmission(raw):
    raw_str = str(raw).strip().lower()
    # If starts with m (like manual, man, m) -> Manual, else default to Automatic
    if raw_str.startswith('m'):
        return 'Manual'
    return 'Automatic'

def normalize_fuel(raw):
    raw_str = str(raw).strip().lower()
    if 'hyb' in raw_str: return 'Hybrid'
    if 'elec' in raw_str or 'ev' in raw_str: return 'Electric'
    if 'dies' in raw_str: return 'Diesel'
    return 'Petrol'

def normalize_variant(raw_variant, model_name, fuel_type):
    var_str = str(raw_variant).strip()
    if not var_str or var_str.lower() == model_name.lower():
        if model_name.lower() == 'vezel':
            return 'Hybrid Z'
        return model_name
    
    # Common mappings
    if model_name.lower() == 'vezel':
        if 'z' in var_str.lower() and 'sensing' in var_str.lower():
            return 'Hybrid Z Honda Sensing'
        if 'z' in var_str.lower():
            return 'Hybrid Z'
        if 'rs' in var_str.lower():
            return 'Hybrid RS Honda Sensing'
        if 'x' in var_str.lower():
            return 'Hybrid X'
        return 'Hybrid Z'
        
    if model_name.lower() == 'swift':
        if 'dlx' in var_str.lower(): return 'DLX 1.3'
        if 'glx' in var_str.lower(): return 'GLX CVT'
        if 'gl' in var_str.lower(): return 'GL CVT'
        
    if model_name.lower() == 'corolla':
        if 'grande' in var_str.lower(): return 'Altis Grande 1.8'
        if 'altis' in var_str.lower() and '1.6' in var_str.lower(): return 'Altis 1.6'
        if 'gli' in var_str.lower(): return 'GLi 1.3 VVTi'
        
    if model_name.lower() == 'civic':
        if 'rs' in var_str.lower() or 'turbo' in var_str.lower(): return '1.5 RS Turbo'
        if 'oriel' in var_str.lower(): return 'Oriel 1.8 i-VTEC'
        
    return var_str.title()

def main():
    print("=" * 65)
    print(">> KAARWALA: FAIR MARKET PRICE PREDICTOR (TERMINAL TEST) <<")
    print("=" * 65)
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "models", "kaarwala_pipeline.joblib"),
        os.path.join(base_dir, "..", "5th_SEMESTER_PROJECTS", "ML-PROJECT", "KaarWala", "models", "kaarwala_pipeline.joblib"),
        r"C:\Users\M Wasay\VS CODE ALL LANG FOLDERS\5th_SEMESTER_PROJECTS\ML-PROJECT\KaarWala\models\kaarwala_pipeline.joblib"
    ]
    
    model_path = None
    for c in candidates:
        if os.path.exists(c):
            model_path = c
            break
            
    lookup_path = os.path.join(os.path.dirname(model_path), "..", "data", "market_lookup.csv") if model_path else None
    
    if not model_path or not os.path.exists(model_path):
        print("[!] Error: Model file not found!")
        return

    print("Loading KaarWala Production Pipeline...")
    model = joblib.load(model_path)
    lookup = pd.read_csv(lookup_path) if lookup_path and os.path.exists(lookup_path) else None
    print("[OK] Model loaded successfully!\n")
    
    print("--- OLX AD DETAILS ENTER KAREIN ---")
    
    # User Inputs with Robust Title-Casing & Normalization
    raw_brand = input("1. Brand (e.g. Toyota, Suzuki, Honda, KIA): ").strip()
    brand = "KIA" if raw_brand.upper() == "KIA" else raw_brand.title()
    
    model_name = input("2. Model (e.g. Corolla, Alto, Civic, Swift, Cultus, Vezel): ").strip().title()
    raw_var = input("3. Variant (e.g. Hybrid Z, DLX 1.3, VXR, Altis 1.6, or press Enter): ").strip()
    
    year = int(input("4. Year (e.g. 2014, 2018, 2021, 2023): ").strip())
    mileage_km = float(input("5. Mileage in KM (e.g. 45000, 69000): ").strip())
    engine_cc = float(input("6. Engine CC (e.g. 660, 1000, 1300, 1500, 1600, EV ke liye 0): ").strip())
    
    raw_fuel = input("7. Fuel Type [Petrol / Hybrid / Diesel / Electric] (Default Petrol): ").strip()
    fuel_type = normalize_fuel(raw_fuel)
    
    variant = normalize_variant(raw_var, model_name, fuel_type)
    
    raw_trans = input("8. Transmission [Manual / Automatic] (Default Automatic): ").strip()
    transmission = normalize_transmission(raw_trans)
    
    city = input("9. City (e.g. Karachi, Lahore, Islamabad, Rawalpindi): ").strip().title()
    if not city:
        city = "Karachi"
        
    asking_price_str = input("10. OLX Seller Asking Price in PKR (e.g. 4250000) [Optional]: ").strip()
    asking_price = float(asking_price_str) if asking_price_str else None

    # Automated Feature Derivations
    vehicle_age = max(2026 - year, 0)
    km_per_year = round(mileage_km / max(vehicle_age, 1), 0)
    
    city_map = {
        'Karachi': 'Sindh', 'Hyderabad': 'Sindh', 'Sukkur': 'Sindh',
        'Lahore': 'Punjab', 'Rawalpindi': 'Punjab', 'Faisalabad': 'Punjab',
        'Multan': 'Punjab', 'Gujranwala': 'Punjab', 'Sialkot': 'Punjab',
        'Islamabad': 'Islamabad', 'Peshawar': 'KPK', 'Quetta': 'Balochistan'
    }
    province = city_map.get(city, 'Sindh' if city == 'Karachi' else 'Punjab')
    
    # Query Market Lookup with exact Year fallback
    market_supply = 50.0
    market_avg_mil = 100000.0
    if lookup is not None:
        lookup_brand = lookup['brand'].astype(str).str.lower().str.strip()
        lookup_model = lookup['model'].astype(str).str.lower().str.strip()
        year_clipped = min(year, 2024)
        
        # Try exact brand + model + year
        exact = lookup[(lookup_brand == brand.lower()) & (lookup_model == model_name.lower()) & (lookup['year'] == year_clipped)]
        if not exact.empty and not np.isnan(exact['market_avg_mileage'].values[0]):
            market_supply = float(exact['market_supply_count'].values[0])
            market_avg_mil = float(exact['market_avg_mileage'].values[0])
        else:
            model_match = lookup[(lookup_brand == brand.lower()) & (lookup_model == model_name.lower())]
            if not model_match.empty:
                market_supply = float(model_match['market_supply_count'].median())
                market_avg_mil = float(model_match['market_avg_mileage'].median())
                
    if np.isnan(market_supply): market_supply = 50.0
    if np.isnan(market_avg_mil) or market_avg_mil <= 0: market_avg_mil = 100000.0
    
    mileage_vs_market = round(mileage_km / market_avg_mil, 2)
    
    # Construct Single-Row DataFrame
    input_df = pd.DataFrame([{
        'brand': brand,
        'model': model_name,
        'variant': variant,
        'year': year,
        'mileage_km': mileage_km,
        'fuel_type': fuel_type,
        'transmission': transmission,
        'engine_cc': 0.0 if fuel_type == 'Electric' else engine_cc,
        'city': city,
        'market_supply_count': market_supply,
        'market_avg_mileage': market_avg_mil,
        'vehicle_age': vehicle_age,
        'km_per_year': km_per_year,
        'province': province,
        'mileage_vs_market': mileage_vs_market
    }])
    
    # Predict Price (Clamped to minimum realistic vehicle price in Pakistan = 1.5 Lakh)
    raw_pred = float(model.predict(input_df)[0])
    predicted_pkr = max(raw_pred, 150000.0)
    pred_lakh = predicted_pkr / 100000.0
    
    # Proportional Fair Range (10% - 15% realistic negotiation window)
    band = max(predicted_pkr * 0.12, 50000.0)
    range_min = max(predicted_pkr - band, 150000.0)
    range_max = predicted_pkr + band
    
    # Display Result
    print("\n" + "=" * 65)
    print(">> PREDICTION RESULT <<")
    print("=" * 65)
    print(f"Vehicle:           {brand} {model_name} {variant} ({year})")
    print(f"Specs:             {mileage_km:,.0f} km | {engine_cc:.0f}cc | {transmission} | {city}")
    print("-" * 65)
    print(f"[*] FAIR MARKET PRICE: PKR {predicted_pkr:,.0f} (approx. {pred_lakh:.2f} Lakh PKR)")
    print(f"[*] FAIR RANGE:        PKR {range_min:,.0f} to PKR {range_max:,.0f} ({range_min/100000:.2f}L - {range_max/100000:.2f}L)")
    
    if asking_price:
        diff = predicted_pkr - asking_price
        diff_pct = (abs(diff) / asking_price) * 100
        accuracy = max(100.0 - diff_pct, 0.0)
        print("-" * 65)
        print(f"Seller Asking Price: PKR {asking_price:,.0f} ({asking_price/100000:.2f} Lakh PKR)")
        if abs(diff) <= band:
            print(f"Verdict:             [FAIR MARKET VALUE] Difference sirf Rs. {abs(diff):,.0f} (~{abs(diff)/100000:.2f} Lakh)")
        elif diff < -band:
            print(f"Verdict:             [OVERPRICED by Seller] Market se Rs. {abs(diff):,.0f} mehngi hai")
        else:
            print(f"Verdict:             [GREAT DEAL / UNDERPRICED] Market se Rs. {abs(diff):,.0f} sasti hai")
        print(f"Match Accuracy:      {accuracy:.1f}%")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
