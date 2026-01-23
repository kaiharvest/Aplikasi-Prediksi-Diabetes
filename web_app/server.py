from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import shap

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Load the trained model, scaler, and other assets
model_path = os.path.join("..", "outputs", "best_model.pkl")
scaler_path = os.path.join("..", "outputs", "scaler.pkl")
results_path = os.path.join("..", "outputs", "summary_results.csv")

# Check if files exist before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
if not os.path.exists(results_path):
    raise FileNotFoundError(f"Results file not found at {results_path}")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Load the best model metrics to determine which features were used
results_df = pd.read_csv(results_path)
best_model_row = results_df.loc[results_df['f1'].idxmax()]
feature_set_name = best_model_row['feature_set']

# Define the feature sets (this should match the sets defined in main.py)
feature_names_original = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

selection_sets = {
    "All": list(range(len(feature_names_original))),
    "RFE": [1, 2, 5, 6, 7],
    "Boruta": [0, 1, 5, 6, 7],
    "GA": [1, 2, 4, 5, 6, 7],
    "PSO": [0, 1, 2, 4, 5, 7],
    "GWO": [0, 1, 2, 4, 5, 7],
    "TopSHAP5": [1, 0, 5, 7, 6]
}

best_feature_indices = selection_sets.get(feature_set_name, list(range(len(feature_names_original))))
best_feature_names = [feature_names_original[i] for i in best_feature_indices]

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/model_info', methods=['GET'])
def get_model_info():
    try:
        # Return the best model information
        model_info = {
            'best_model': f"{feature_set_name}-{best_model_row['model']}",
            'feature_set': feature_set_name,
            'accuracy': float(best_model_row['accuracy']),
            'f1_score': float(best_model_row['f1']),
            'precision': float(best_model_row['precision']),
            'recall': float(best_model_row['recall']),
            'auc': float(best_model_row.get('roc_auc', 0.0))
        }

        return jsonify(model_info)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.json

        # Extract values
        pregnancies = data.get('pregnancies', 0)
        glucose = data.get('glucose', 0)
        blood_pressure = data.get('bloodPressure', 0)
        skin_thickness = data.get('skinThickness', 0)
        insulin = data.get('insulin', 0)
        bmi = data.get('bmi', 0)
        dpf = data.get('dpf', 0)
        age = data.get('age', 0)

        # Create input dataframe
        feature_names_original = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                                 columns=feature_names_original)

        # Scale the input
        input_data_scaled = scaler.transform(input_data)

        # Select only the features used by the best model
        final_input = input_data_scaled[:, best_feature_indices]

        # Make prediction
        prediction = model.predict(final_input)[0]
        prediction_proba = model.predict_proba(final_input)[0]

        # Calculate SHAP values for the prediction
        shap_df = pd.DataFrame(final_input, columns=best_feature_names)
        shap_values = explainer.shap_values(shap_df)

        # Handle different formats of SHAP output
        if isinstance(shap_values, list):
            # For binary classification, take SHAP values for positive class (index 1)
            shap_values_for_prediction = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            # For other cases
            if len(shap_values.shape) == 3:
                shap_values_for_prediction = shap_values[0, :, 1]  # Take first sample, all features, positive class
            else:
                shap_values_for_prediction = shap_values[0]  # Take first sample

        # Return the results
        result = {
            'prediction': int(prediction),
            'probability': float(prediction_proba[1]),  # Probability of positive class
            'shap_values': shap_values_for_prediction.tolist(),
            'feature_names': best_feature_names,
            'feature_set_used': feature_set_name
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)