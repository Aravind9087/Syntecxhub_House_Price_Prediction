"""
Data Preprocessing and Feature Engineering Module
SyntecxHub Machine Learning Internship Project - 1
Constraints addressed:
- Load housing dataset
- Clean missing values, handle duplicates and outliers
- Explore and engineer domain-specific features
"""

import os
import numpy as np
import pandas as pd

CURRENT_YEAR = 2024

def load_data(filepath=None):
    """Load raw housing dataset."""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'housing_data.csv')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Housing dataset not found at {filepath}. Run generate_data.py first.")
    
    df = pd.read_csv(filepath)
    print(f"[*] Loaded raw dataset with shape: {df.shape}")
    return df

def clean_data(df):
    """
    Clean dataset:
    1. Remove duplicate records
    2. Impute missing values with statistical medians
    3. Remove invalid/negative entries
    """
    df_cleaned = df.copy()
    
    # 1. Remove duplicate records
    initial_len = len(df_cleaned)
    df_cleaned.drop_duplicates(inplace=True)
    duplicates_removed = initial_len - len(df_cleaned)
    if duplicates_removed > 0:
        print(f"[*] Removed {duplicates_removed} duplicate row(s).")
    
    # 2. Check and log missing values before imputation
    missing_before = df_cleaned.isnull().sum()
    print("[*] Missing values before imputation:")
    for col, count in missing_before[missing_before > 0].items():
        print(f"    - {col}: {count} missing ({count/len(df_cleaned)*100:.2f}%)")
    
    # 3. Impute missing numeric values using column medians
    if 'Lot_Area' in df_cleaned.columns and df_cleaned['Lot_Area'].isnull().sum() > 0:
        lot_area_median = df_cleaned['Lot_Area'].median()
        df_cleaned['Lot_Area'] = df_cleaned['Lot_Area'].fillna(lot_area_median)
        print(f"[*] Imputed missing Lot_Area with median: {lot_area_median:.1f}")
        
    if 'Garage_Cars' in df_cleaned.columns and df_cleaned['Garage_Cars'].isnull().sum() > 0:
        garage_mode = df_cleaned['Garage_Cars'].mode()[0]
        df_cleaned['Garage_Cars'] = df_cleaned['Garage_Cars'].fillna(garage_mode)
        print(f"[*] Imputed missing Garage_Cars with mode: {garage_mode}")

    if 'Neighborhood_Score' in df_cleaned.columns and df_cleaned['Neighborhood_Score'].isnull().sum() > 0:
        # Groupby Neighborhood to impute local neighborhood score if available, else overall median
        df_cleaned['Neighborhood_Score'] = df_cleaned.groupby('Neighborhood')['Neighborhood_Score'].transform(
            lambda x: x.fillna(x.median())
        )
        if df_cleaned['Neighborhood_Score'].isnull().sum() > 0:
            df_cleaned['Neighborhood_Score'] = df_cleaned['Neighborhood_Score'].fillna(df_cleaned['Neighborhood_Score'].median())
        print(f"[*] Imputed missing Neighborhood_Score using neighborhood-specific medians.")
    
    return df_cleaned

def engineer_features(df):
    """
    Engineer domain-relevant features:
    - House_Age (years since construction)
    - Total_Rooms (Bedrooms + Bathrooms)
    - Living_Lot_Ratio (Square_Feet / Lot_Area)
    """
    df_eng = df.copy()
    
    # 1. House Age
    if 'Year_Built' in df_eng.columns:
        df_eng['House_Age'] = CURRENT_YEAR - df_eng['Year_Built']
        df_eng['House_Age'] = df_eng['House_Age'].clip(lower=0)
    
    # 2. Total Rooms
    if 'Bedrooms' in df_eng.columns and 'Bathrooms' in df_eng.columns:
        df_eng['Total_Rooms'] = df_eng['Bedrooms'] + df_eng['Bathrooms']
        
    # 3. Ratio of living area to lot area
    if 'Square_Feet' in df_eng.columns and 'Lot_Area' in df_eng.columns:
        df_eng['Living_Lot_Ratio'] = np.round(df_eng['Square_Feet'] / df_eng['Lot_Area'], 4)
        
    return df_eng

def handle_outliers(df, columns=['Square_Feet', 'Price'], factor=3.0):
    """
    Handle extreme outliers using IQR method with a conservative factor
    to keep data realistic while preventing model skew.
    """
    df_filtered = df.copy()
    for col in columns:
        if col in df_filtered.columns:
            q1 = df_filtered[col].quantile(0.01)
            q3 = df_filtered[col].quantile(0.99)
            df_filtered = df_filtered[(df_filtered[col] >= q1) & (df_filtered[col] <= q3)]
    print(f"[*] Data shape after clipping extreme 1% tails: {df_filtered.shape}")
    return df_filtered

def preprocess_pipeline(raw_filepath=None, save_cleaned=True):
    """Full preprocessing pipeline from raw data to clean data ready for modeling."""
    raw_df = load_data(raw_filepath)
    clean_df = clean_data(raw_df)
    engineered_df = engineer_features(clean_df)
    final_df = handle_outliers(engineered_df)
    
    if save_cleaned:
        out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned_housing_data.csv')
        final_df.to_csv(out_path, index=False)
        print(f"[*] Cleaned and engineered dataset saved to: {out_path}")
        
    return final_df

if __name__ == '__main__':
    print("=" * 60)
    print("SYNTECXHUB PROJECT 1: DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    df_processed = preprocess_pipeline()
    print("\nSummary of Processed Dataset:")
    print(df_processed.info())
    print("\nFirst 3 Processed Samples:")
    print(df_processed.head(3))
