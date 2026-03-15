from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load Model
with open('health_model.pkl', 'rb') as f:
    data = pickle.load(f)
    model, all_symptoms = data['model'], data['symptoms']

# Categorization
HISTORY = ['extra_marital_contacts', 'history_of_alcohol_consumption', 'family_history', 'receiving_blood_transfusion', 'receiving_unsterile_injections']
LAB_SIGNS = ['irregular_sugar_level', 'dark_urine', 'yellow_urine', 'polyuria', 'fluid_overload', 'fast_heart_rate', 'blood_in_sputum']

@app.route('/')
def index():
    cat_history = [s for s in all_symptoms if s in HISTORY]
    cat_lab = [s for s in all_symptoms if s in LAB_SIGNS]
    cat_physical = [s for s in all_symptoms if s not in HISTORY and s not in LAB_SIGNS]
    return render_template('index.html', physical=cat_physical, history=cat_history, lab=cat_lab)

@app.route('/get_consultation', methods=['POST'])
def get_consultation():
    selected = request.form.getlist('symptoms')
    if not selected:
        return "Please select at least one symptom."

    # Binary Vectorization
    input_vec = np.zeros(len(all_symptoms))
    for s in selected:
        if s in all_symptoms:
            input_vec[all_symptoms.index(s)] = 1
            
    # Top-3 Inference
    probs = model.predict_proba([input_vec])[0]
    top_3_idx = np.argsort(probs)[-3:][::-1]
    
    desc_df = pd.read_csv('dataset/symptom_Description.csv')
    results = []

    for idx in top_3_idx:
        disease = model.classes_[idx]
        desc = desc_df[desc_df['Disease'].str.strip() == disease.strip()]['Description'].values
        results.append({
            'disease': disease,
            'confidence': round(probs[idx] * 100, 2),
            'description': desc[0] if len(desc) > 0 else "Details not available."
        })

    return render_template('result.html', results=results)

if __name__ == '__main__':
    # Use port from environment variable for deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
