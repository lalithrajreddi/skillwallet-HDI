from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Paths
MODEL_PATH = "hdi_model.pkl"
DATA_PATH = "hdi_processed.csv"

# Global references to be loaded
model = None
df_countries = None

def load_resources():
    global model, df_countries
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            print("Successfully loaded machine learning model.")
        except Exception as e:
            print(f"Error loading model: {e}")
            
    if os.path.exists(DATA_PATH):
        try:
            df_countries = pd.read_csv(DATA_PATH, keep_default_na=False)
            print(f"Successfully loaded country database ({len(df_countries)} countries).")
        except Exception as e:
            print(f"Error loading country database: {e}")

# Load model and data immediately
load_resources()

# Category thresholds
def categorize_hdi(score):
    if score >= 0.800:
        return 'Very High'
    elif score >= 0.700:
        return 'High'
    elif score >= 0.550:
        return 'Medium'
    else:
        return 'Low'

@app.route('/')
def home():
    # Renders our modern single-page dashboard
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model, df_countries
    
    # Reload if they weren't loaded properly
    if model is None or df_countries is None:
        load_resources()
        if model is None:
            return jsonify({'error': 'Prediction model is not trained/loaded yet.'}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Extract features and convert to floats
        try:
            life_expectancy = float(data.get('life_expectancy'))
            expected_schooling = float(data.get('expected_schooling'))
            mean_schooling = float(data.get('mean_schooling'))
            gni_capita = float(data.get('gni_capita'))
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid features. All inputs must be numeric values.'}), 400

        # Validate input ranges to ensure physical realism
        errors = []
        if not (20.0 <= life_expectancy <= 100.0):
            errors.append("Life expectancy must be between 20 and 100 years.")
        if not (0.0 <= expected_schooling <= 25.0):
            errors.append("Expected years of schooling must be between 0 and 25 years.")
        if not (0.0 <= mean_schooling <= 20.0):
            errors.append("Mean years of schooling must be between 0 and 20 years.")
        if not (100.0 <= gni_capita <= 150000.0):
            errors.append("GNI per capita must be between 100 and 150,000 PPP$.")
            
        if errors:
            return jsonify({'error': 'Validation Error', 'details': errors}), 400

        # Step 1: Model Prediction (using Linear Regression)
        features = [[life_expectancy, expected_schooling, mean_schooling, gni_capita]]
        predicted_hdi = float(model.predict(features)[0])
        
        # Bounding HDI score to [0, 1] as per index rules
        predicted_hdi = max(0.0, min(1.0, predicted_hdi))
        predicted_hdi = round(predicted_hdi, 3)
        
        # Classify Development Tier
        tier = categorize_hdi(predicted_hdi)

        # Step 2: Compute True Dimensions (for educational/reference comparisons)
        # Health Index = (Life expectancy - 20) / (85 - 20)
        true_health = max(0.0, min(1.0, (life_expectancy - 20) / 65))
        
        # Education Index = (Expected_Schooling_Index + Mean_Schooling_Index) / 2
        exp_index = max(0.0, min(1.0, expected_schooling / 18))
        mean_index = max(0.0, min(1.0, mean_schooling / 15))
        true_edu = (exp_index + mean_index) / 2
        
        # Income Index = (ln(GNI_cleaned) - ln(100)) / (ln(75000) - ln(100))
        gni_cleaned = max(100.0, min(75000.0, gni_capita))
        true_inc = (np.log(gni_cleaned) - np.log(100.0)) / (np.log(75000.0) - np.log(100.0))
        true_inc = max(0.0, min(1.0, true_inc))
        
        # Standard Formula-based HDI
        formula_hdi = round((true_health * true_edu * true_inc) ** (1/3), 3)

        # Step 3: Find Comparable Countries
        comparisons = []
        if df_countries is not None:
            # Calculate absolute distance in HDI score
            df_temp = df_countries.copy()
            df_temp['distance'] = (df_temp['HDI Score'] - predicted_hdi).abs()
            # Sort by distance and retrieve top 3 closest countries
            df_closest = df_temp.sort_values(by='distance').head(3)
            for _, row in df_closest.iterrows():
                comparisons.append({
                    'country': row['Country Name'],
                    'hdi': float(row['HDI Score']),
                    'tier': row['Development Tier'],
                    'life_expectancy': float(row['Life expectancy']),
                    'gni': float(row['GNI per capita'])
                })

        return jsonify({
            'predicted_hdi': predicted_hdi,
            'tier': tier,
            'formula_hdi': formula_hdi,
            'indices': {
                'health': round(true_health, 3),
                'education': round(true_edu, 3),
                'income': round(true_inc, 3)
            },
            'comparisons': comparisons
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/countries', methods=['GET'])
def get_countries():
    global df_countries
    if df_countries is None:
        load_resources()
        if df_countries is None:
            return jsonify({'error': 'Country database is not loaded yet.'}), 500
            
    search = request.args.get('search', '').lower()
    tier = request.args.get('tier', '')
    
    filtered_df = df_countries.copy()
    
    if search:
        filtered_df = filtered_df[filtered_df['Country Name'].str.lower().str.contains(search)]
        
    if tier:
        filtered_df = filtered_df[filtered_df['Development Tier'] == tier]
        
    # Standard output formatting
    records = []
    for _, row in filtered_df.iterrows():
        records.append({
            'rank': int(row['Rank']) if 'Rank' in row else None,
            'country': row['Country Name'],
            'iso': row['Country ISO'] if 'Country ISO' in row else '',
            'life_expectancy': float(row['Life expectancy']),
            'expected_schooling': float(row['Expected yrs of schooling']),
            'mean_schooling': float(row['Mean yrs of schooling']),
            'gni': float(row['GNI per capita']),
            'hdi': float(row['HDI Score']),
            'tier': row['Development Tier']
        })
        
    return jsonify(records)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    global df_countries
    if df_countries is None:
        load_resources()
        if df_countries is None:
            return jsonify({'error': 'Country database is not loaded yet.'}), 500
            
    # Count of countries by Development Tier
    tier_counts = df_countries['Development Tier'].value_counts().to_dict()
    
    # Global averages
    global_averages = {
        'hdi': float(df_countries['HDI Score'].mean()),
        'life_expectancy': float(df_countries['Life expectancy'].mean()),
        'expected_schooling': float(df_countries['Expected yrs of schooling'].mean()),
        'mean_schooling': float(df_countries['Mean yrs of schooling'].mean()),
        'gni': float(df_countries['GNI per capita'].mean())
    }
    
    # Tier-specific stats
    tier_averages = {}
    for tier in ['Very High', 'High', 'Medium', 'Low']:
        df_tier = df_countries[df_countries['Development Tier'] == tier]
        if not df_tier.empty:
            tier_averages[tier] = {
                'count': len(df_tier),
                'hdi': float(df_tier['HDI Score'].mean()),
                'life_expectancy': float(df_tier['Life expectancy'].mean()),
                'expected_schooling': float(df_tier['Expected yrs of schooling'].mean()),
                'mean_schooling': float(df_tier['Mean yrs of schooling'].mean()),
                'gni': float(df_tier['GNI per capita'].mean())
            }
        else:
            tier_averages[tier] = {
                'count': 0, 'hdi': 0.0, 'life_expectancy': 0.0, 
                'expected_schooling': 0.0, 'mean_schooling': 0.0, 'gni': 0.0
            }
            
    model_info = {}
    if model is not None:
        try:
            model_info = {
                'coefficients': {
                    'life_expectancy': float(model.coef_[0]),
                    'expected_schooling': float(model.coef_[1]),
                    'mean_schooling': float(model.coef_[2]),
                    'gni_capita': float(model.coef_[3])
                },
                'intercept': float(model.intercept_)
            }
        except Exception as e:
            print(f"Error compiling model info: {e}")
            
    return jsonify({
        'total_countries': len(df_countries),
        'tier_counts': tier_counts,
        'global_averages': global_averages,
        'tier_averages': tier_averages,
        'model_info': model_info
    })

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
