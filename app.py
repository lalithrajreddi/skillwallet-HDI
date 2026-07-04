from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Paths
MODEL_PATH = "hdi_model.pkl"
DATA_PATH = "hdi_processed.csv"
ENCODER_PATH = "country_encoder.pkl"

# Global references to be loaded
model = None
df_countries = None
encoder = None

def load_resources():
    global model, df_countries, encoder
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

    if os.path.exists(ENCODER_PATH):
        try:
            with open(ENCODER_PATH, 'rb') as f:
                encoder = pickle.load(f)
            print("Successfully loaded country label encoder.")
        except Exception as e:
            print(f"Error loading encoder: {e}")

# Load model, data, and encoder immediately
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
    # Renders the new home page (home.html)
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global model, df_countries, encoder
    
    # Reload if they weren't loaded properly
    if model is None or df_countries is None or encoder is None:
        load_resources()
        if model is None:
            if request.is_json:
                return jsonify({'error': 'Prediction model is not trained/loaded yet.'}), 500
            else:
                return "Error: Model not loaded."

    # Sort country list alphabetically for dropdown selection
    countries_list = []
    if df_countries is not None:
        countries_list = sorted(df_countries['Country Name'].unique().tolist())

    # GET request: Render the indexnew.html form page
    if request.method == 'GET':
        return render_template('indexnew.html', countries=countries_list)

    # POST request: support BOTH JSON (AJAX) and HTML Form Post
    if request.is_json:
        # ------------------ JSON AJAX Flow ------------------
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No input data provided'}), 400

            # Extract features
            try:
                life_expectancy = float(data.get('life_expectancy'))
                expected_schooling = float(data.get('expected_schooling'))
                mean_schooling = float(data.get('mean_schooling'))
                gni_capita = float(data.get('gni_capita'))
                country_name = data.get('country_name', countries_list[0] if countries_list else 'Norway')
            except (ValueError, TypeError):
                return jsonify({'error': 'Invalid features. All inputs must be numeric values.'}), 400

            # Validate input ranges
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

            # Label Encode country name
            try:
                country_encoded = int(encoder.transform([country_name])[0])
            except Exception:
                country_encoded = 0

            # Model Prediction (using Linear Regression with the 5 variables)
            features = [[country_encoded, life_expectancy, expected_schooling, mean_schooling, gni_capita]]
            predicted_hdi = float(model.predict(features)[0])
            predicted_hdi = max(0.0, min(1.0, predicted_hdi))
            predicted_hdi = round(predicted_hdi, 3)
            
            tier = categorize_hdi(predicted_hdi)

            # Compute True Dimension Indices
            true_health = max(0.0, min(1.0, (life_expectancy - 20) / 65))
            exp_index = max(0.0, min(1.0, expected_schooling / 18))
            mean_index = max(0.0, min(1.0, mean_schooling / 15))
            true_edu = (exp_index + mean_index) / 2
            gni_cleaned = max(100.0, min(75000.0, gni_capita))
            true_inc = (np.log(gni_cleaned) - np.log(100.0)) / (np.log(75000.0) - np.log(100.0))
            true_inc = max(0.0, min(1.0, true_inc))
            formula_hdi = round((true_health * true_edu * true_inc) ** (1/3), 3)

            # Find Comparable Countries
            comparisons = []
            if df_countries is not None:
                df_temp = df_countries.copy()
                df_temp['distance'] = (df_temp['HDI Score'] - predicted_hdi).abs()
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

    else:
        # ------------------ HTML Form Post Flow ------------------
        try:
            country_name = request.form.get('country_name')
            life_expectancy_str = request.form.get('life_expectancy')
            expected_schooling_str = request.form.get('expected_schooling')
            mean_schooling_str = request.form.get('mean_schooling')
            gni_capita_str = request.form.get('gni_capita')

            if not all([country_name, life_expectancy_str, expected_schooling_str, mean_schooling_str, gni_capita_str]):
                return render_template('indexnew.html', countries=countries_list, error="Please fill all indicators.")

            life_expectancy = float(life_expectancy_str)
            expected_schooling = float(expected_schooling_str)
            mean_schooling = float(mean_schooling_str)
            gni_capita = float(gni_capita_str)

            # Label Encode country name
            try:
                country_encoded = int(encoder.transform([country_name])[0])
            except Exception:
                country_encoded = 0

            # Model Prediction (using Linear Regression with the 5 variables)
            features = [[country_encoded, life_expectancy, expected_schooling, mean_schooling, gni_capita]]
            predicted_hdi = float(model.predict(features)[0])
            predicted_hdi = max(0.0, min(1.0, predicted_hdi))
            predicted_hdi = round(predicted_hdi, 3)
            
            tier = categorize_hdi(predicted_hdi)

            # Compute Dimension Indices
            true_health = max(0.0, min(1.0, (life_expectancy - 20) / 65.0))
            exp_index = max(0.0, min(1.0, expected_schooling / 18.0))
            mean_index = max(0.0, min(1.0, mean_schooling / 15.0))
            true_edu = (exp_index + mean_index) / 2.0
            gni_cleaned = max(100.0, min(75000.0, gni_capita))
            true_inc = (np.log(gni_cleaned) - np.log(100.0)) / (np.log(75000.0) - np.log(100.0))
            true_inc = max(0.0, min(1.0, true_inc))

            # Find Comparable Countries
            comparisons = []
            if df_countries is not None:
                df_temp = df_countries.copy()
                df_temp['distance'] = (df_temp['HDI Score'] - predicted_hdi).abs()
                df_closest = df_temp.sort_values(by='distance').head(3)
                for _, row in df_closest.iterrows():
                    comparisons.append({
                        'country': row['Country Name'],
                        'hdi': float(row['HDI Score']),
                        'tier': row['Development Tier'],
                        'life_expectancy': float(row['Life expectancy']),
                        'gni': float(row['GNI per capita'])
                    })

            prediction_result = {
                'predicted_hdi': predicted_hdi,
                'tier': tier,
                'indices': {
                    'health': round(true_health, 3),
                    'education': round(true_edu, 3),
                    'income': round(true_inc, 3)
                },
                'comparisons': comparisons
            }

            return render_template('indexnew.html', 
                                   countries=countries_list,
                                   selected_country=country_name,
                                   life_expectancy=life_expectancy,
                                   expected_schooling=expected_schooling,
                                   mean_schooling=mean_schooling,
                                   gni_capita=gni_capita,
                                   prediction=prediction_result)

        except Exception as e:
            return render_template('indexnew.html', countries=countries_list, error=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
