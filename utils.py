import pandas as pd
import numpy as np

def standardize_districts(df):
    """Standardizes casing and uses 'starts-with' logic to fix typos/homoglyphs."""
    if 'district' not in df.columns:
        return df
    
    # 1. Standardize casing and remove hidden spaces
    df['district'] = df['district'].str.strip().str.capitalize()

    # 2. Robust fix for typos (e.g., Burepa, Gakenki, Musanza)
    # This matches the logic from your final notebook
    df.loc[df['district'].str.startswith('Bur', na=False), 'district'] = 'Burera'
    df.loc[df['district'].str.startswith('Nya', na=False), 'district'] = 'Nyabihu'
    df.loc[df['district'].str.startswith('Gak', na=False), 'district'] = 'Gakenke'
    df.loc[df['district'].str.startswith('Mus', na=False), 'district'] = 'Musanze'
    df.loc[df['district'].str.startswith('Rul', na=False), 'district'] = 'Rulindo'
    
    return df

def fix_spatial_data(df):
    """Detects and fixes swapped Latitude/Longitude coordinates for Rwanda."""
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # In Rwanda, Lat is negative and Lon is positive. Swap if incorrect.
        mask = df['latitude'] > df['longitude']
        df.loc[mask, ['latitude', 'longitude']] = df.loc[mask, ['longitude', 'latitude']].values
    return df

def impute_localized_and_clip(df):
    """Performs district-level imputation and clips fuel outliers at 99th percentile."""
    # 1. District-level Imputation
    geo_cols = ['elevation_m', 'latitude', 'longitude', 'distance_to_market_km', 'household_size']
    for col in geo_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df.groupby('district')[col].transform('median'))
    
    # 2. Outlier Clipping (As seen in version2_final)
    if 'baseline_fuel_kg_person_week' in df.columns:
        upper_bound = df['baseline_fuel_kg_person_week'].quantile(0.99)
        df['baseline_fuel_kg_person_week'] = df['baseline_fuel_kg_person_week'].clip(upper=upper_bound)
        
    return df

def engineer_features(df):
    """Creates Market Climb Index, Seasonality, and Remoteness features."""
    # 1. Market Climb Index
    if 'distance_to_market_km' in df.columns and 'elevation_m' in df.columns:
        df['market_climb_index'] = df['distance_to_market_km'] * (df['elevation_m'] / 1000)
    
    # 2. Seasonality (dayfirst=True to match notebook)
    if 'distribution_date' in df.columns:
        df['distribution_date'] = pd.to_datetime(df['distribution_date'], dayfirst=True, errors='coerce')
        df['dist_month'] = df['distribution_date'].dt.month
        df['is_rainy_season'] = df['dist_month'].isin([3, 4, 5, 9, 10, 11, 12]).astype(int)
    
    # 3. Relative Remoteness
    if 'latitude' in df.columns and 'longitude' in df.columns:
        centers = df.groupby('district')[['latitude', 'longitude']].transform('mean')
        df['relative_remoteness'] = np.sqrt(
            (df['latitude'] - centers['latitude'])**2 + 
            (df['longitude'] - centers['longitude'])**2
        )
    
    return df

def calculate_target(df):
    """Calculates pct_reduction and defines low_adoption target (< 30%)."""
    usage_cols = [f'usage_month_{i}' for i in range(1, 7)]
    if not all(col in df.columns for col in usage_cols + ['baseline_fuel_kg_person_week']):
        return df

    # Mask negative values
    df[usage_cols] = df[usage_cols].mask(df[usage_cols] < 0)

    # Calculation logic
    df['weekly_baseline_hh'] = df['baseline_fuel_kg_person_week'] * df['household_size']
    df['avg_weekly_usage'] = df[usage_cols].mean(axis=1)
    df['pct_reduction'] = (df['weekly_baseline_hh'] - df['avg_weekly_usage']) / df['weekly_baseline_hh']
    
    # Target definition
    df['low_adoption'] = (df['pct_reduction'] < 0.30).astype(int)
    
    return df.dropna(subset=['low_adoption'])

def full_pipeline(filepath_or_df):
    """Orchestrates the entire cleaning and engineering process."""
    df = pd.read_csv(filepath_or_df) if isinstance(filepath_or_df, str) else filepath_or_df.copy()
    df = standardize_districts(df)
    df = fix_spatial_data(df)
    df = impute_localized_and_clip(df)
    df = engineer_features(df)
    df = calculate_target(df)
    return df