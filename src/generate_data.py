"""
Data Generation Script for House Price Prediction
SyntecxHub Machine Learning Internship Project - 1
Generates a realistic synthetic housing dataset with genuine market relationships,
realistic noise, missing values, and categorical features for cleaning demonstration.
"""

import os
import numpy as np
import pandas as pd

def generate_housing_data(n_samples=1500, random_seed=42):
    np.random.seed(random_seed)
    
    # 1. Base Features
    # Square feet between 700 and 4200
    sqft = np.random.normal(loc=1950, scale=650, size=n_samples)
    sqft = np.clip(sqft, 650, 4800).round()
    
    # Bedrooms correlated with square feet (1 to 5)
    bedrooms = np.clip(np.round(sqft / 550 + np.random.normal(0, 0.6, n_samples)), 1, 6).astype(int)
    
    # Bathrooms correlated with bedrooms and sqft (1 to 4)
    bathrooms = np.clip(np.round(bedrooms * 0.65 + (sqft / 1200) + np.random.normal(0, 0.35, n_samples) * 0.5) * 0.5 * 2, 1.0, 4.5)
    bathrooms = np.round(bathrooms, 1)
    
    # Year Built (1950 to 2023)
    year_built = np.random.randint(1955, 2024, size=n_samples)
    
    # Garage Cars (0 to 3)
    garage_cars = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.08, 0.27, 0.50, 0.15])
    
    # Lot Area in sqft (2500 to 18000)
    lot_area = np.clip(np.random.normal(loc=8500, scale=2800, size=n_samples) + sqft * 1.5, 2000, 22000).round()
    
    # Neighborhoods with different quality / premium tiers
    neighborhoods = np.random.choice(
        ['Suburban Heights', 'Downtown Central', 'Oakwood Hills', 'Riverside Park', 'Greenway Estates'],
        size=n_samples,
        p=[0.25, 0.20, 0.22, 0.18, 0.15]
    )
    
    neighborhood_multipliers = {
        'Downtown Central': 1.25,
        'Oakwood Hills': 1.15,
        'Greenway Estates': 1.05,
        'Riverside Park': 0.95,
        'Suburban Heights': 0.88
    }
    
    # Overall Quality rating (1 to 10 scale, median around 6-7)
    overall_quality = np.clip(np.round(np.random.normal(6.5, 1.4, size=n_samples)), 2, 10).astype(int)
    
    # Neighborhood Quality Score (1 to 10 scale)
    neighborhood_scores = np.array([
        {'Downtown Central': 9.2, 'Oakwood Hills': 8.5, 'Greenway Estates': 7.2, 'Riverside Park': 6.0, 'Suburban Heights': 5.2}[n]
        for n in neighborhoods
    ])
    neighborhood_scores += np.random.normal(0, 0.3, size=n_samples)
    neighborhood_scores = np.clip(np.round(neighborhood_scores, 1), 1.0, 10.0)

    # 2. Target Variable Calculation (True Real-Estate Economics)
    # Base Price: $45,000
    # + $125 per sqft
    # + $18,000 per bedroom
    # + $22,000 per bathroom
    # + $15,000 per garage car
    # + $3.50 per lot area sqft
    # + $24,000 per overall quality point
    # + $12,000 per neighborhood score point
    # - $650 per year of age
    current_year = 2024
    age = current_year - year_built
    
    noise = np.random.normal(0, 18000, size=n_samples)
    
    price = (
        45000 +
        (sqft * 130) +
        (bedrooms * 16500) +
        (bathrooms * 21000) +
        (garage_cars * 14500) +
        (lot_area * 3.2) +
        (overall_quality * 23500) +
        (neighborhood_scores * 11500) -
        (age * 680) +
        noise
    )
    
    # Apply neighborhood multiplier lightly
    price = np.round(price, -2)
    price = np.maximum(price, 60000) # minimum price floor
    
    # 3. Create DataFrame
    df = pd.DataFrame({
        'Square_Feet': sqft,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Year_Built': year_built,
        'Garage_Cars': garage_cars,
        'Lot_Area': lot_area,
        'Neighborhood': neighborhoods,
        'Overall_Quality': overall_quality,
        'Neighborhood_Score': neighborhood_scores,
        'Price': price
    })
    
    # 4. Inject realistic missing values and outlier anomalies for data cleaning demonstration
    # 3% missing in Lot_Area
    missing_lot_indices = np.random.choice(n_samples, size=int(n_samples * 0.035), replace=False)
    df.loc[missing_lot_indices, 'Lot_Area'] = np.nan
    
    # 2% missing in Garage_Cars
    missing_garage_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    df.loc[missing_garage_indices, 'Garage_Cars'] = np.nan
    
    # 1.5% missing in Neighborhood_Score
    missing_neigh_indices = np.random.choice(n_samples, size=int(n_samples * 0.015), replace=False)
    df.loc[missing_neigh_indices, 'Neighborhood_Score'] = np.nan

    # Add a few duplicates to clean (e.g., 5 duplicate rows)
    duplicates = df.iloc[:5].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    return df

if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, 'housing_data.csv')
    
    print("Generating realistic Housing Dataset for SyntecxHub Project 1...")
    df = generate_housing_data(n_samples=1600, random_seed=42)
    df.to_csv(raw_path, index=False)
    print(f"Dataset successfully saved to: {raw_path}")
    print(f"Dataset Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values per column:")
    print(df.isnull().sum())
