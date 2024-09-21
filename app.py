from flask import Flask, render_template, request, send_from_directory
import joblib
import numpy as np
import os
import json

app = Flask(
    __name__,
    template_folder='.',    # Current directory for templates
    static_folder='.',      # Current directory for static files
    static_url_path='/static'  # URL prefix for static files
)

# Load the trained model
model = joblib.load('disease_model.pkl')

# Define the mapping from numeric labels to disease names
disease_mapping = {
    0: 'Fungal infection',
    1: 'Allergy',
    2: 'GERD',
    3: 'Chronic cholestasis',
    4: 'Drug Reaction',
    5: 'Peptic ulcer disease',
    6: 'AIDS',
    7: 'Diabetes',
    8: 'Gastroenteritis',
    9: 'Bronchial Asthma',
    10: 'Hypertension',
    11: 'Migraine',
    12: 'Cervical spondylosis',
    13: 'Paralysis (brain hemorrhage)',
    14: 'Jaundice',
    15: 'Malaria',
    16: 'Chicken pox',
    17: 'Dengue',
    18: 'Typhoid',
    19: 'Hepatitis A',
    20: 'Hepatitis B',
    21: 'Hepatitis C',
    22: 'Hepatitis D',
    23: 'Hepatitis E',
    24: 'Alcoholic hepatitis',
    25: 'Tuberculosis',
    26: 'Common Cold',
    27: 'Pneumonia',
    28: 'Dimorphic hemorrhoids (piles)',
    29: 'Heart attack',
    30: 'Varicose veins',
    31: 'Hypothyroidism',
    32: 'Hyperthyroidism',
    33: 'Hypoglycemia',
    34: 'Osteoarthrosis',
    35: 'Arthritis',
    36: '(Vertigo) Paroxysmal Positional Vertigo',
    37: 'Acne',
    38: 'Urinary tract infection',
    39: 'Psoriasis',
    40: 'Impetigo'
    # Add more mappings as per your dataset
}

# Route for the home page (index.html)
@app.route('/')
def home():
    return render_template('index.html')  # Your main HTML file

# Route for the About page (about.html)
@app.route('/about')
def about():
    return render_template('about.html')

# Route for the Contact Us page (contact_us.html)
@app.route('/contact_us')
def contact_us():
    return render_template('contact_us.html')

# Route to handle prediction (predict.html)
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # List of features in the same order as your model expects
        feature_names = [
            'itching', 'continuous_sneezing', 'joint_pain', 'stomach_pain',
            'acidity', 'ulcers_on_tongue', 'anxiety', 'irregular_sugar_level',
            'cough', 'dehydration', 'headache', 'yellowish_skin', 'dark_urine',
            'nausea', 'pain_behind_the_eyes', 'diarrhoea', 'mild_fever',
            'blurred_and_distorted_vision', 'redness_of_eyes', 'runny_nose',
            'chest_pain', 'fast_heart_rate', 'bloody_stool', 'cramps',
            'obesity', 'enlarged_thyroid', 'red_spots_over_body',
            'abnormal_menstruation', 'receiving_blood_transfusion',
            'receiving_unsterile_injections', 'history_of_alcohol_consumption'
        ]

        input_data = []
        for feature in feature_names:
            value = request.form.get(feature)
            if value == 'yes':
                input_data.append(1)
            else:
                input_data.append(0)

        # Convert to numpy array and reshape for the model
        input_array = np.array(input_data).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_array)
        numeric_result = prediction[0]

        # Map the numeric result to the disease name
        disease_name = disease_mapping.get(numeric_result, "Unknown disease")

        return render_template('result.html', prediction=disease_name)
    else:
        return render_template('predict.html')  # Render the predict form on GET request

# Route to serve static files (CSS and Images)
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(debug=True)
