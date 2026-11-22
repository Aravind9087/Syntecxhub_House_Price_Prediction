"""
End-to-End Master Pipeline Runner
SyntecxHub Machine Learning Internship Project - 1: House Price Prediction
Runs the complete lifecycle:
1. Dataset Generation / Ingestion
2. Data Preprocessing & Cleaning
3. Exploratory Data Analysis & Visualizations
4. Feature Selection & Linear Regression Model Training
5. Model Evaluation (R², RMSE, MAE) & Coefficient Interpretation
6. Model Serialization & Example Inferences
"""

import sys
import os
import time

def banner():
    print("=" * 80)
    print("  SYNTECXHUB INTERNSHIP PROGRAM - MACHINE LEARNING WEEK 1")
    print("  PROJECT 1: HOUSE PRICE PREDICTION (LINEAR REGRESSION)")
    print("=" * 80)

def run():
    banner()
    start_time = time.time()
    
    # Step 1: Data Generation / Loading
    print("\n[STEP 1/5] GENERATING HOUSING DATASET...")
    from src.generate_data import generate_housing_data
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    raw_path = os.path.join(data_dir, 'housing_data.csv')
    df_raw = generate_housing_data(n_samples=1600, random_seed=42)
    df_raw.to_csv(raw_path, index=False)
    print(f"-> Generated {df_raw.shape[0]} raw samples saved to {raw_path}")
    
    # Step 2: Data Preprocessing & Cleaning
    print("\n[STEP 2/5] CLEANING & FEATURE ENGINEERING...")
    from src.data_preprocessing import preprocess_pipeline
    df_clean = preprocess_pipeline(raw_path, save_cleaned=True)
    print(f"-> Cleaned dataset contains {df_clean.shape[0]} samples and {df_clean.shape[1]} columns")

    # Step 3: Exploratory Data Analysis
    print("\n[STEP 3/5] EXPLORATORY DATA ANALYSIS & VISUALIZATION...")
    from src.eda import perform_eda
    plots_dir = os.path.join(os.path.dirname(__file__), 'static', 'plots')
    perform_eda(output_plots_dir=plots_dir)

    # Step 4: Model Training, Evaluation, and Coefficient Interpretation
    print("\n[STEP 4/5] TRAINING LINEAR REGRESSION MODEL...")
    from src.train import train_and_evaluate
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    model, metadata = train_and_evaluate(model_dir=model_dir, plots_dir=plots_dir)

    # Step 5: Example Predictions on Saved Model
    print("\n[STEP 5/5] RUNNING EXAMPLE INFERENCES ON SAVED MODEL...")
    from src.predict import run_example_predictions
    run_example_predictions()

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print(f"Saved Model   : {os.path.join(model_dir, 'house_price_model.pkl')}")
    print(f"Saved Plots   : {plots_dir}")
    print(f"Flask Web App : Run 'python app.py' to launch interactive interface")
    print("=" * 80)

if __name__ == '__main__':
    run()
