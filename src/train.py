"""
Model Training and Evaluation Pipeline
SyntecxHub Machine Learning Internship Project - 1
Constraints addressed:
- Feature selection
- Train / Test split (80/20)
- Train Linear Regression model
- Evaluate performance using RMSE and R²
- Interpret regression coefficients and intercept
- Save model and evaluation artifacts for production deployment
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import matplotlib
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(__file__), '..', '.mpl_cache')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

SELECTED_FEATURES = [
    'Square_Feet',
    'Bedrooms',
    'Bathrooms',
    'House_Age',
    'Garage_Cars',
    'Lot_Area',
    'Overall_Quality',
    'Neighborhood_Score'
]
TARGET_COL = 'Price'

def train_and_evaluate(data_path=None, model_dir=None, plots_dir=None):
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned_housing_data.csv')
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    if plots_dir is None:
        plots_dir = os.path.join(os.path.dirname(__file__), '..', 'static', 'plots')

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Load Cleaned Dataset
    print(f"[*] Loading preprocessed data from {data_path}...")
    df = pd.read_csv(data_path)
    
    X = df[SELECTED_FEATURES]
    y = df[TARGET_COL]
    
    print(f"[*] Selected Features ({len(SELECTED_FEATURES)}): {SELECTED_FEATURES}")
    print(f"[*] Target Variable: {TARGET_COL}")
    print(f"[*] Dataset Size: {X.shape[0]} samples")
    
    # 2. Train / Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"[*] Train set size: {X_train.shape[0]} samples (80%)")
    print(f"[*] Test set size : {X_test.shape[0]} samples (20%)")
    
    # 3. Fit Linear Regression Model
    print("[*] Training Ordinary Least Squares (OLS) Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 4. Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 5. Model Evaluation Metrics
    # In scikit-learn 1.4+, mean_squared_error squared=False is deprecated in favor of root_mean_squared_error
    try:
        from sklearn.metrics import root_mean_squared_error
        train_rmse = root_mean_squared_error(y_train, y_train_pred)
        test_rmse = root_mean_squared_error(y_test, y_test_pred)
    except ImportError:
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    train_mape = mean_absolute_percentage_error(y_train, y_train_pred) * 100
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
    
    print("\n" + "=" * 65)
    print("MODEL EVALUATION RESULTS (Linear Regression)")
    print("=" * 65)
    print(f"{'Metric':<25} | {'Training Set':<18} | {'Testing Set':<18}")
    print("-" * 65)
    print(f"{'R-squared (R²)':<25} | {train_r2:.4f}{'':<12} | {test_r2:.4f}")
    print(f"{'RMSE ($ USD)':<25} | ${train_rmse:,.2f}{'':<6} | ${test_rmse:,.2f}")
    print(f"{'MAE ($ USD)':<25} | ${train_mae:,.2f}{'':<7} | ${test_mae:,.2f}")
    print(f"{'MAPE (%)':<25} | {train_mape:.2f}%{'':<13} | {test_mape:.2f}%")
    print("=" * 65)
    
    # 6. Coefficient Analysis & Interpretation
    print("\n" + "=" * 65)
    print("COEFFICIENT INTERPRETATION & MARGINAL EFFECTS")
    print("=" * 65)
    print(f"Intercept (Base Price β₀): ${model.intercept_:,.2f}")
    print("-" * 65)
    print(f"{'Feature Name':<20} | {'Coefficient (β)':<16} | {'Unit Interpretation'}")
    print("-" * 65)
    
    coeff_dict = {}
    for feat, coef in zip(SELECTED_FEATURES, model.coef_):
        coeff_dict[feat] = float(coef)
        sign = "+" if coef >= 0 else "-"
        print(f"{feat:<20} | {sign}${abs(coef):>12,.2f} | For each +1 unit, Price changes by {sign}${abs(coef):,.2f}")
    print("=" * 65)

    # Standardized Coefficients for relative importance ranking
    std_x = X_train.std(axis=0)
    std_y = y_train.std()
    standardized_coefs = model.coef_ * (std_x / std_y)
    
    importance_df = pd.DataFrame({
        'Feature': SELECTED_FEATURES,
        'Coefficient': model.coef_,
        'Standardized_Coeff': standardized_coefs,
        'Abs_Standardized': np.abs(standardized_coefs)
    }).sort_values('Abs_Standardized', ascending=False)
    
    print("\nTop 5 Features Ranked by Impact (Standardized Beta):")
    for idx, row in importance_df.head(5).iterrows():
        print(f"  {row['Feature']:<20}: Std Beta = {row['Standardized_Coeff']:+.4f} (Raw: ${row['Coefficient']:+,.2f})")
    
    # 7. Generate Evaluation Plots
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Actual vs Predicted Prices on Test Set
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_test_pred, alpha=0.55, color='#2563eb', edgecolors='none', label='Test Observations')
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color='#dc2626', linestyle='--', linewidth=2, label='Perfect Prediction (y = ŷ)')
    ax.set_title(f'Actual vs. Predicted House Prices (Test R² = {test_r2:.3f})', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Actual Price ($ USD)', fontsize=11)
    ax.set_ylabel('Predicted Price ($ USD)', fontsize=11)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"${x*1e-3:.0f}k"))
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"${x*1e-3:.0f}k"))
    ax.legend(loc='upper left')
    plt.tight_layout()
    act_pred_path = os.path.join(plots_dir, 'actual_vs_predicted.png')
    fig.savefig(act_pred_path, dpi=150)
    plt.close()
    print(f"[*] Saved Actual vs Predicted plot to: {act_pred_path}")
    
    # Plot 2: Residuals Distribution
    residuals = y_test - y_test_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color='#0891b2', bins=30, ax=ax, edgecolor='white')
    ax.axvline(0, color='#dc2626', linestyle='--', linewidth=1.5, label='Zero Error')
    ax.set_title('Residuals Distribution on Test Set', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Residual Error (Actual - Predicted) ($ USD)', fontsize=11)
    ax.set_ylabel('Density / Count', fontsize=11)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"${x*1e-3:.0f}k"))
    ax.legend()
    plt.tight_layout()
    res_path = os.path.join(plots_dir, 'residuals_distribution.png')
    fig.savefig(res_path, dpi=150)
    plt.close()
    print(f"[*] Saved Residuals plot to: {res_path}")

    # Plot 3: Feature Coefficients (Importance)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#16a34a' if c >= 0 else '#dc2626' for c in importance_df['Coefficient']]
    ax.barh(importance_df['Feature'], importance_df['Coefficient'], color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.8)
    ax.set_title('Linear Regression Model Feature Coefficients (Dollar Impact)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Marginal Price Impact ($ USD per unit)', fontsize=11)
    ax.invert_yaxis()
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"${x*1e-3:.0f}k"))
    plt.tight_layout()
    feat_path = os.path.join(plots_dir, 'feature_importance.png')
    fig.savefig(feat_path, dpi=150)
    plt.close()
    print(f"[*] Saved Feature Importance plot to: {feat_path}")
    
    # 8. Save Trained Model & Metadata
    model_save_path = os.path.join(model_dir, 'house_price_model.pkl')
    joblib.dump(model, model_save_path)
    print(f"[*] Successfully saved model to: {model_save_path}")
    
    metadata = {
        'model_type': 'LinearRegression',
        'intercept': float(model.intercept_),
        'features': SELECTED_FEATURES,
        'coefficients': coeff_dict,
        'metrics': {
            'train_r2': round(float(train_r2), 4),
            'test_r2': round(float(test_r2), 4),
            'train_rmse': round(float(train_rmse), 2),
            'test_rmse': round(float(test_rmse), 2),
            'train_mae': round(float(train_mae), 2),
            'test_mae': round(float(test_mae), 2),
            'train_mape': round(float(train_mape), 2),
            'test_mape': round(float(test_mape), 2)
        },
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'standardized_importance': importance_df[['Feature', 'Standardized_Coeff', 'Coefficient']].to_dict(orient='records')
    }
    
    meta_path = os.path.join(model_dir, 'model_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"[*] Successfully saved model metadata to: {meta_path}")
    
    return model, metadata

if __name__ == '__main__':
    train_and_evaluate()
