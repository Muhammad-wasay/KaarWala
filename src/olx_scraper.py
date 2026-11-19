"""
KaarWala: OLX Pakistan Car Listings Extractor & Pipeline Processor
Author: Mind Masters (CT-24236, CT-24238, CT-24075)
Purpose: Collect live used-car ads from OLX Pakistan (olx.com.pk)
         and transform them into the exact 15-feature matrix for
         independent external test evaluation.
"""

import os
import re
import pandas as pd
import numpy as np

# City to Province mapping for Pakistani car market
CITY_TO_PROVINCE = {
    'Karachi': 'Sindh', 'Hyderabad': 'Sindh', 'Sukkur': 'Sindh',
    'Lahore': 'Punjab', 'Rawalpindi': 'Punjab', 'Faisalabad': 'Punjab',
    'Multan': 'Punjab', 'Gujranwala': 'Punjab', 'Sialkot': 'Punjab',
    'Sargodha': 'Punjab', 'Bahawalpur': 'Punjab', 'Sheikhupura': 'Punjab',
    'Islamabad': 'Islamabad',
    'Peshawar': 'KPK', 'Abbottabad': 'KPK', 'Mardan': 'KPK',
    'Quetta': 'Balochistan'
}

def clean_price(price_str):
    """Clean asking price string to numeric PKR value."""
    if isinstance(price_str, (int, float)):
        return int(price_str)
    cleaned = re.sub(r'[^\d]', '', str(price_str))
    return int(cleaned) if cleaned else 0

def clean_mileage(mileage_str):
    """Clean mileage string to float km."""
    if isinstance(mileage_str, (int, float)):
        return float(mileage_str)
    cleaned = re.sub(r'[^\d]', '', str(mileage_str))
    return float(cleaned) if cleaned else np.nan

def process_olx_listings(raw_records, lookup_path="../data/market_lookup.csv"):
    """
    Takes raw OLX listings (Brand, Model, Variant, Year, Mileage, Fuel, Transmission, Engine CC, City, Price)
    and computes the 6 engineered features required by the KaarWala ML Pipeline:
    1. vehicle_age = 2026 - year
    2. km_per_year = mileage_km / vehicle_age
    3. province = derived from city
    4. market_supply_count = fetched from market_lookup.csv
    5. market_avg_mileage = fetched from market_lookup.csv
    6. mileage_vs_market = mileage_km / market_avg_mileage
    """
    df = pd.DataFrame(raw_records)
    
    # Clean numeric fields
    df['price_in_pkr'] = df['price_in_pkr'].apply(clean_price)
    df['mileage_km'] = df['mileage_km'].apply(clean_mileage)
    df['year'] = df['year'].astype(int)
    
    # Handle EV Engine Displacement (Domain Rule from Member 2)
    df.loc[df['fuel_type'] == 'Electric', 'engine_cc'] = 0.0
    
    # Load market lookup
    if os.path.exists(lookup_path):
        lookup = pd.read_csv(lookup_path)
        lookup['brand_k'] = lookup['brand'].astype(str).str.lower().str.strip()
        lookup['model_k'] = lookup['model'].astype(str).str.lower().str.strip()
        lookup['year_k'] = lookup['year'].astype(int)
        
        df['brand_k'] = df['brand'].astype(str).str.lower().str.strip()
        df['model_k'] = df['model'].astype(str).str.lower().str.strip()
        df['year_k'] = df['year'].clip(upper=2024)
        
        df = df.merge(
            lookup[['brand_k', 'model_k', 'year_k', 'market_supply_count', 'market_avg_mileage']].drop_duplicates(),
            on=['brand_k', 'model_k', 'year_k'],
            how='left'
        )
        df.drop(columns=['brand_k', 'model_k', 'year_k'], inplace=True)
    else:
        df['market_supply_count'] = np.nan
        df['market_avg_mileage'] = np.nan

    # Fill lookup fallbacks
    df['market_supply_count'] = df['market_supply_count'].fillna(df['market_supply_count'].median()).fillna(50.0).round(0)
    df['market_avg_mileage'] = df['market_avg_mileage'].fillna(df['market_avg_mileage'].median()).fillna(65000.0).round(0)
    
    # Feature engineering
    df['vehicle_age'] = 2026 - df['year']
    df['km_per_year'] = (df['mileage_km'] / df['vehicle_age'].clip(lower=1)).round(0)
    df['province'] = df['city'].map(CITY_TO_PROVINCE).fillna('Punjab')
    df['mileage_vs_market'] = (df['mileage_km'] / df['market_avg_mileage']).round(2)
    
    # Ensure exact 16 columns order
    ordered_cols = [
        'brand', 'model', 'variant', 'year', 'mileage_km', 'fuel_type',
        'transmission', 'engine_cc', 'city', 'market_supply_count',
        'market_avg_mileage', 'vehicle_age', 'km_per_year', 'province',
        'price_in_pkr', 'mileage_vs_market'
    ]
    return df[ordered_cols]

if __name__ == "__main__":
    test_path = os.path.join(os.path.dirname(__file__), "../data/olx_test_cars.csv")
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path)
        print(f"Loaded {len(df_test)} verified OLX listings from {test_path}")
        print(df_test[['brand', 'model', 'year', 'city', 'price_in_pkr']].head())
    else:
        print("OLX test dataset not found.")
