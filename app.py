"""
Flask Web Application for House Price Prediction
SyntecxHub Machine Learning Internship Project - 1
Provides an interactive web interface for real-time house valuation and model inspection.
"""

import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'house_price_model.pkl')
META_PATH = os.path.join(BASE_DIR, 'models', 'model_metadata.json')

# Load trained model and metadata
model = None
metadata = {}

def get_model():
    global model, metadata
    if model is None:
        if not os.path.exists(MODEL_PATH):
            from run_pipeline import run
            run()
        model = joblib.load(MODEL_PATH)
        if os.path.exists(META_PATH):
            with open(META_PATH, 'r') as f:
                metadata = json.load(f)
    return model, metadata

@app.route('/', methods=['GET'])
def index():
    _, meta = get_model()
    # Default values for initial form load
    default_inputs = {
        'Square_Feet': 2200,
        'Bedrooms': 3,
        'Bathrooms': 2.5,
        'House_Age': 10,
        'Garage_Cars': 2,
        'Lot_Area': 8500,
        'Overall_Quality': 7,
        'Neighborhood_Score': 7.5
    }
    return render_template(
        'index.html',
        meta=meta,
        defaults=default_inputs,
        prediction=None,
        breakdown=None
    )

@app.route('/predict', methods=['POST'])
def predict():
    mdl, meta = get_model()
    features = meta.get('features', [
        'Square_Feet', 'Bedrooms', 'Bathrooms', 'House_Age',
        'Garage_Cars', 'Lot_Area', 'Overall_Quality', 'Neighborhood_Score'
    ])
    
    try:
        # Check if JSON request or Form submit
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        # Parse inputs
        inputs = {
            'Square_Feet': float(data.get('Square_Feet', 2000)),
            'Bedrooms': int(float(data.get('Bedrooms', 3))),
            'Bathrooms': float(data.get('Bathrooms', 2.0)),
            'House_Age': int(float(data.get('House_Age', 10))),
            'Garage_Cars': int(float(data.get('Garage_Cars', 2))),
            'Lot_Area': float(data.get('Lot_Area', 8000)),
            'Overall_Quality': int(float(data.get('Overall_Quality', 7))),
            'Neighborhood_Score': float(data.get('Neighborhood_Score', 7.0))
        }

        # Predict
        df_input = pd.DataFrame([[inputs[f] for f in features]], columns=features)
        pred_val = mdl.predict(df_input)[0]
        
        # Calculate feature contributions
        contributions = {'Base Price (β₀)': round(float(mdl.intercept_), 2)}
        for feat, coef in zip(features, mdl.coef_):
            contributions[feat] = round(float(inputs[feat] * coef), 2)

        if request.is_json:
            return jsonify({
                'status': 'success',
                'predicted_price': round(float(pred_val), 2),
                'formatted_price': f"${pred_val:,.2f}",
                'inputs': inputs,
                'contributions': contributions
            })
            
        return render_template(
            'index.html',
            meta=meta,
            defaults=inputs,
            prediction=round(float(pred_val), 2),
            formatted_price=f"${pred_val:,.2f}",
            contributions=contributions
        )

    except Exception as e:
        if request.is_json:
            return jsonify({'status': 'error', 'message': str(e)}), 400
        return render_template('index.html', meta=meta, defaults={}, error=str(e))

if __name__ == '__main__':
    # Initialize model
    get_model()
    print("[*] Starting SyntecxHub House Price Prediction Server on http://127.0.0.1:5000 ...")
    app.run(host='127.0.0.1', port=5000, debug=False)
