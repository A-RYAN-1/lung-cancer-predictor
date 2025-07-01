from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import requests
from dotenv import load_dotenv
import os
import logging

# Initialize Flask app
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Hugging Face API settings
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HF_API_URL = "https://api-inference.huggingface.co/models/mixtralai/Mixtral-8x7B-Instruct-v0.1"

# Load trained model and scaler
model = joblib.load('./models/lung_cancer_model.pkl')
scaler = joblib.load('./models/scaler.pkl')

# Define input fields expected by the model
FIELDS = [
    'Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy',
    'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease',
    'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker',
    'Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss',
    'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty',
    'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect and process form data
        form_data = {field: float(request.form.get(field)) for field in FIELDS}
        # Convert to DataFrame to preserve feature names
        input_df = pd.DataFrame([form_data], columns=FIELDS)

        # Scale input
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)[0]
        label_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
        risk_level = label_map.get(prediction, "Unknown")
        logger.info(f"Prediction: {risk_level}")

        # Generate prompt for LLM
        symptoms_summary = "\n".join([
            f"- {field}: {'Yes' if val == 1 else 'No'}" if field not in ['Age', 'Gender']
            else f"- {field}: {int(val)}"
            for field, val in form_data.items()
        ])

        prompt = f"""
[INST] You are a helpful medical assistant AI providing general information, not professional medical advice. A user has provided the following health profile:

{symptoms_summary}

A machine learning model has predicted their lung cancer risk to be: **{risk_level}**.

Provide a concise, friendly explanation (2-3 sentences) of why this might be the case, focusing on key risk factors (e.g., smoking, air pollution). Suggest 1-2 specific, practical lifestyle changes or precautions (e.g., quitting smoking, consulting a doctor). [/INST]
"""

        # Call Hugging Face API
        try:
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            payload = {
                "inputs": prompt,
                "parameters": {"max_new_tokens": 150, "temperature": 0.7, "return_full_text": False}
            }
            response = requests.post(HF_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            explanation = response.json()[0]["generated_text"].strip()
            logger.info("Hugging Face API call successful")
        except Exception as api_error:
            logger.error(f"Hugging Face API error: {str(api_error)}")
            # Mock response for testing
            explanation = f"Your {risk_level.lower()} risk is likely due to {'smoking and air pollution' if form_data['Smoking'] == 1 or form_data['Air Pollution'] == 1 else 'minimal risk factors'}. Consider {'quitting smoking' if form_data['Smoking'] == 1 else 'maintaining a healthy lifestyle'} and consulting a healthcare provider."

        return render_template('index.html',
                               prediction_text=f"Predicted Lung Cancer Risk: {risk_level}",
                               explanation_text=explanation)

    except Exception as e:
        logger.error(f"General error: {str(e)}")
        return render_template('index.html',
                               prediction_text=f"⚠️ Error: {str(e)}",
                               explanation_text="Unable to generate explanation due to an error.")

if __name__ == "__main__":
    app.run(debug=True)