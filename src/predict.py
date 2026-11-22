"""
Model Inference and Example Predictions Module
SyntecxHub Machine Learning Internship Project - 1
Constraints addressed:
- Load saved model
- Show example predictions on novel unseen house samples
- Provide feature contribution breakdown and comparison tables
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

def load_saved_model(model_path=None, meta_path=None):
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'house_price_model.pkl')
    if meta_path is None:
        meta_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_metadata.json')
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Saved model not found at {model_path}. Run train.py first.")
        
    model = joblib.load(model_path)
    
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            
    return model, metadata

def predict_single(house_dict, model, features):
    """Predict price for a single house dictionary and return breakdown."""
    row = [house_dict[f] for f in features]
    df_single = pd.DataFrame([row], columns=features)
    pred_price = model.predict(df_single)[0]
    
    # Feature contribution breakdown
    breakdown = {'Base Intercept (β₀)': model.intercept_}
    for feat, coef in zip(features, model.coef_):
        breakdown[feat] = house_dict[feat] * coef
        
    return pred_price, breakdown

def run_example_predictions():
    model, meta = load_saved_model()
    features = meta.get('features', [
        'Square_Feet', 'Bedrooms', 'Bathrooms', 'House_Age',
        'Garage_Cars', 'Lot_Area', 'Overall_Quality', 'Neighborhood_Score'
    ])
    
    print("=" * 80)
    print("SYNTECXHUB PROJECT 1: HOUSE PRICE PREDICTION - EXAMPLE INFERENCES")
    print("=" * 80)
    print(f"Loaded Model: LinearRegression (Trained on {meta.get('train_samples', 'N/A')} records)")
    print(f"Validation Test R²: {meta.get('metrics', {}).get('test_r2', 'N/A')} | Test RMSE: ${meta.get('metrics', {}).get('test_rmse', 'N/A'):,}")
    print("=" * 80)

    # 4 Real-World Archetype House Profiles
    archetypes = [
        {
            'Profile': '1. Urban Starter Home',
            'Square_Feet': 1100,
            'Bedrooms': 2,
            'Bathrooms': 1.0,
            'House_Age': 20,
            'Garage_Cars': 1,
            'Lot_Area': 4500,
            'Overall_Quality': 5,
            'Neighborhood_Score': 6.0
        },
        {
            'Profile': '2. Suburban Family Residence',
            'Square_Feet': 2250,
            'Bedrooms': 4,
            'Bathrooms': 2.5,
            'House_Age': 12,
            'Garage_Cars': 2,
            'Lot_Area': 9500,
            'Overall_Quality': 7,
            'Neighborhood_Score': 7.5
        },
        {
            'Profile': '3. Modern Luxury Villa',
            'Square_Feet': 3650,
            'Bedrooms': 5,
            'Bathrooms': 4.0,
            'House_Age': 2,
            'Garage_Cars': 3,
            'Lot_Area': 16000,
            'Overall_Quality': 9,
            'Neighborhood_Score': 9.2
        },
        {
            'Profile': '4. Historic Vintage Cottage',
            'Square_Feet': 1600,
            'Bedrooms': 3,
            'Bathrooms': 1.5,
            'House_Age': 58,
            'Garage_Cars': 1,
            'Lot_Area': 6200,
            'Overall_Quality': 6,
            'Neighborhood_Score': 6.8
        }
    ]
    
    results = []
    for house in archetypes:
        profile_name = house['Profile']
        pred_price, breakdown = predict_single(house, model, features)
        results.append({
            'Profile': profile_name,
            'SqFt': house['Square_Feet'],
            'Beds': house['Bedrooms'],
            'Baths': house['Bathrooms'],
            'Age': f"{house['House_Age']} yrs",
            'Garage': f"{house['Garage_Cars']} car",
            'Quality': f"{house['Overall_Quality']}/10",
            'Neighb.': f"{house['Neighborhood_Score']}/10",
            'Predicted Price': f"${pred_price:,.2f}"
        })
        
    res_df = pd.DataFrame(results)
    print("\nSummary Comparison Table of Example Predictions:")
    print(res_df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("DETAILED FEATURE-BY-FEATURE ECONOMIC BREAKDOWN:")
    print("=" * 80)
    for house in archetypes:
        profile_name = house['Profile']
        pred_price, breakdown = predict_single(house, model, features)
        print(f"\n>>> {profile_name} (Total Predicted Price: ${pred_price:,.2f})")
        for comp, val in breakdown.items():
            sign = "+" if val >= 0 else "-"
            print(f"    - {comp:<24}: {sign}${abs(val):>10,.2f}")

    return res_df

if __name__ == '__main__':
    run_example_predictions()
