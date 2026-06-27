# ============================================================
# AROGYA: Edge AI Diagnostic Website
# Flask Backend
# ============================================================

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
import warnings
import os
import json
from datetime import datetime
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── GLOBAL MODEL VARIABLES ───────────────────────────────────
model          = None
scaler         = None
model_anemia   = None
scaler_anemia  = None
model_heart    = None
scaler_heart   = None
model_ht       = None
scaler_ht      = None
model_dg       = None
model_maln     = None
scaler_maln    = None
models_loaded  = False

# ── LIVE SENSOR DATA (from ESP32) ─────────────────────────────
latest_sensor_reading = {
    'pulse':       None,
    'spo2':        None,
    'temperature': None,
    'updated_at':  None
}

iso_forest     = None
normal_avg     = None
disease_names  = ['Diabetes','Anemia','Heart','Hypertension',
                  'Malnutrition','Dengue']
alert_multipliers = {
    'Diabetes':2.0,'Anemia':2.5,'Heart':2.0,
    'Hypertension':2.0,'Malnutrition':2.5,'Dengue':3.0}

# ── LOAD ALL MODELS ON STARTUP ───────────────────────────────
def load_all_models():
    global model, scaler, model_anemia, scaler_anemia
    global model_heart, scaler_heart, model_ht, scaler_ht
    global model_dg, model_maln, scaler_maln, models_loaded
    global iso_forest, normal_avg

    print("Loading AROGYA models...")

    # 1. DIABETES
    try:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                'Insulin','BMI','DiabetesPedigree','Age','Outcome']
        df_pima = pd.read_csv(url, names=cols)
        X_d = df_pima[['Glucose','BMI','Age','Insulin','BloodPressure']].values
        y_d = df_pima['Outcome'].values
        X_tr,X_te,y_tr,y_te = train_test_split(X_d,y_d,test_size=0.2,random_state=42,stratify=y_d)
        scaler = StandardScaler()
        model  = MLPClassifier(hidden_layer_sizes=(16,8),activation='relu',
                               solver='adam',max_iter=1000,random_state=42,early_stopping=True)
        model.fit(scaler.fit_transform(X_tr), y_tr)
        print("Diabetes: Trained on Real Data ✅")
    except Exception as e:
        print("Diabetes GitHub Download FAILED. Using Synthetic Fallback...")
        np.random.seed(42)
        n = 150
        diab_pos = pd.DataFrame({'Glucose': np.random.normal(160, 30, n).clip(126, 300), 'BMI': np.random.normal(32, 5, n).clip(25, 50), 'Age': np.random.normal(55, 12, n).clip(30, 80), 'Insulin': np.random.normal(150, 50, n).clip(50, 400), 'BloodPressure': np.random.normal(90, 15, n).clip(80, 120), 'Outcome': 1})
        diab_neg = pd.DataFrame({'Glucose': np.random.normal(95, 15, n).clip(70, 120), 'BMI': np.random.normal(23, 3, n).clip(18, 25), 'Age': np.random.normal(30, 10, n).clip(18, 60), 'Insulin': np.random.normal(60, 20, n).clip(15, 100), 'BloodPressure': np.random.normal(70, 10, n).clip(60, 80), 'Outcome': 0})
        df_d = pd.concat([diab_pos, diab_neg]).sample(frac=1, random_state=42)
        X_d = df_d[['Glucose','BMI','Age','Insulin','BloodPressure']].values
        y_d = df_d['Outcome'].values
        X_tr,X_te,y_tr,y_te = train_test_split(X_d,y_d,test_size=0.2,random_state=42,stratify=y_d)
        scaler = StandardScaler()
        model  = MLPClassifier(hidden_layer_sizes=(16,8),activation='relu', solver='adam',max_iter=1000,random_state=42,early_stopping=True)
        model.fit(scaler.fit_transform(X_tr), y_tr)

    # 2. ANEMIA
    try:
        np.random.seed(42)
        n = 150
        anemia_pos = pd.DataFrame({'Gender': np.random.choice([0,1], n), 'Hemoglobin': np.random.normal(9.0, 1.5, n).clip(5, 11.9), 'MCH':  np.random.normal(22, 3, n).clip(15, 27), 'MCHC': np.random.normal(30, 2, n).clip(25, 33), 'MCV':  np.random.normal(72, 8, n).clip(55, 79), 'Result': 1})
        anemia_neg = pd.DataFrame({'Gender': np.random.choice([0,1], n), 'Hemoglobin': np.random.normal(14.0, 1.5, n).clip(12, 18), 'MCH':  np.random.normal(29, 2, n).clip(27, 35), 'MCHC': np.random.normal(34, 1.5, n).clip(33, 38), 'MCV':  np.random.normal(88, 5, n).clip(80, 100), 'Result': 0})
        df_anemia = pd.concat([anemia_pos, anemia_neg]).sample(frac=1, random_state=42)
        X_a = df_anemia[['Gender','Hemoglobin','MCH','MCHC','MCV']].values
        y_a = df_anemia['Result'].values
        X_tr,X_te,y_tr,y_te = train_test_split(X_a,y_a,test_size=0.2,random_state=42,stratify=y_a)
        scaler_anemia = StandardScaler()
        model_anemia  = MLPClassifier(hidden_layer_sizes=(16,8),activation='relu', solver='adam',max_iter=1000,random_state=42,early_stopping=True)
        model_anemia.fit(scaler_anemia.fit_transform(X_tr), y_tr)
        print("Anemia: Trained ✅")
    except: pass

    # 3. HEART
    try:
        np.random.seed(42)
        n = 150
        heart_pos = pd.DataFrame({'Age': np.random.normal(58, 10, n).clip(30, 80), 'Systolic_BP': np.random.normal(155, 20, n).clip(120, 200), 'Cholesterol': np.random.normal(240, 30, n).clip(180, 320), 'Diabetes': np.random.choice([0,1], n, p=[0.4,0.6]), 'Hypertension': np.random.choice([0,1], n, p=[0.3,0.7]), 'Smoking': np.random.choice([0,1], n, p=[0.4,0.6]), 'Obesity': np.random.choice([0,1], n, p=[0.4,0.6]), 'Family_History': np.random.choice([0,1], n, p=[0.3,0.7]), 'Gender': np.random.choice([0,1], n, p=[0.3,0.7]), 'Physical_Activity': np.random.choice([0,1,2], n, p=[0.5,0.3,0.2]), 'Result': 1})
        heart_neg = pd.DataFrame({'Age': np.random.normal(38, 12, n).clip(18, 65), 'Systolic_BP': np.random.normal(118, 12, n).clip(90, 140), 'Cholesterol': np.random.normal(185, 25, n).clip(140, 220), 'Diabetes': np.random.choice([0,1], n, p=[0.8,0.2]), 'Hypertension': np.random.choice([0,1], n, p=[0.8,0.2]), 'Smoking': np.random.choice([0,1], n, p=[0.7,0.3]), 'Obesity': np.random.choice([0,1], n, p=[0.7,0.3]), 'Family_History': np.random.choice([0,1], n, p=[0.6,0.4]), 'Gender': np.random.choice([0,1], n, p=[0.5,0.5]), 'Physical_Activity': np.random.choice([0,1,2], n, p=[0.2,0.4,0.4]), 'Result': 0})
        df_heart = pd.concat([heart_pos, heart_neg]).sample(frac=1, random_state=42)
        X_h = df_heart[['Age','Systolic_BP','Cholesterol','Diabetes','Hypertension', 'Smoking','Obesity','Family_History','Gender','Physical_Activity']].values
        y_h = df_heart['Result'].values
        X_tr,X_te,y_tr,y_te = train_test_split(X_h,y_h,test_size=0.2,random_state=42)
        scaler_heart = StandardScaler()
        model_heart  = MLPClassifier(hidden_layer_sizes=(64,32,16),activation='relu', solver='adam',max_iter=1000,random_state=42,early_stopping=True)
        model_heart.fit(scaler_heart.fit_transform(X_tr), y_tr)
        print("Heart: Trained ✅")
    except: pass

    # 4. HYPERTENSION
    try:
        np.random.seed(42)
        n = 150
        ht_pos = pd.DataFrame({'Age': np.random.normal(55, 12, n).clip(30, 80), 'BMI': np.random.normal(28, 4, n).clip(18, 40), 'Systolic_BP': np.random.normal(158, 15, n).clip(140, 200), 'Diastolic_BP': np.random.normal(95, 10, n).clip(90, 120), 'Smoking': np.random.choice([0,1], n, p=[0.5,0.5]), 'Pulse': np.random.normal(88, 12, n).clip(60, 120), 'Result': 1})
        ht_neg = pd.DataFrame({'Age': np.random.normal(35, 10, n).clip(18, 60), 'BMI': np.random.normal(23, 3, n).clip(18, 30), 'Systolic_BP': np.random.normal(115, 10, n).clip(90, 130), 'Diastolic_BP': np.random.normal(75, 8, n).clip(60, 85), 'Smoking': np.random.choice([0,1], n, p=[0.7,0.3]), 'Pulse': np.random.normal(72, 10, n).clip(55, 95), 'Result': 0})
        df_ht = pd.concat([ht_pos, ht_neg]).sample(frac=1, random_state=42)
        feat_cols_ht = ['Age','BMI','Systolic_BP','Diastolic_BP','Smoking','Pulse']
        X_ht = df_ht[feat_cols_ht].values
        y_ht = df_ht['Result'].values
        X_tr,X_te,y_tr,y_te = train_test_split(X_ht,y_ht,test_size=0.2,random_state=42)
        scaler_ht = StandardScaler()
        model_ht  = MLPClassifier(hidden_layer_sizes=(32,16),activation='relu', solver='adam',max_iter=1000,random_state=42,early_stopping=True)
        model_ht.fit(scaler_ht.fit_transform(X_tr), y_tr)
        print("Hypertension: Trained ✅")
    except: pass

    # 5. DENGUE
    try:
        np.random.seed(42)
        n = 150
        dengue_pos = pd.DataFrame({'Age': np.random.normal(28,12,n).clip(5,70).astype(int), 'Gender': np.random.choice([0,1],n), 'Temperature_F': np.random.normal(103.2,1.2,n).clip(100.4,106), 'Heart_Rate': np.random.normal(102,12,n).clip(80,135), 'SpO2': np.random.normal(96,1.5,n).clip(91,99), 'Headache': np.random.choice([0,1],n,p=[0.1,0.9]), 'Body_Ache': np.random.choice([0,1],n,p=[0.08,0.92]), 'Rash': np.random.choice([0,1],n,p=[0.3,0.7]), 'Result': 1})
        dengue_neg = pd.DataFrame({'Age': np.random.normal(33,15,n).clip(5,75).astype(int), 'Gender': np.random.choice([0,1],n), 'Temperature_F': np.random.normal(99.8,1.3,n).clip(97.5,103), 'Heart_Rate': np.random.normal(85,10,n).clip(60,110), 'SpO2': np.random.normal(98,0.8,n).clip(95,100), 'Headache': np.random.choice([0,1],n,p=[0.55,0.45]), 'Body_Ache': np.random.choice([0,1],n,p=[0.55,0.45]), 'Rash': np.random.choice([0,1],n,p=[0.88,0.12]), 'Result': 0})
        df_dg = pd.concat([dengue_pos,dengue_neg]).sample(frac=1,random_state=42)
        X_dg = df_dg[['Age','Gender','Temperature_F','Heart_Rate', 'SpO2','Headache','Body_Ache','Rash']].values
        y_dg = df_dg['Result'].values
        model_dg = RandomForestClassifier(n_estimators=100, random_state=42)
        model_dg.fit(X_dg, y_dg)
        print("Dengue: Trained ✅")
    except: pass

    # 6. MALNUTRITION
    try:
        np.random.seed(42)
        n = 150
        normal_m = pd.DataFrame({'Age': np.random.normal(30,15,n).clip(1,75).astype(int), 'Gender': np.random.choice([0,1],n), 'BMI': np.random.normal(23,3,n).clip(18.5,35), 'Weight_kg': np.random.normal(62,12,n).clip(35,100), 'Height_cm': np.random.normal(165,10,n).clip(100,190), 'Appetite_Loss': np.random.choice([0,1],n,p=[0.9,0.1]), 'Fatigue': np.random.choice([0,1],n,p=[0.85,0.15]), 'Hair_Loss': np.random.choice([0,1],n,p=[0.9,0.1]), 'Pale_Skin': np.random.choice([0,1],n,p=[0.9,0.1]), 'Result': 0})
        maln_m = pd.DataFrame({'Age': np.random.normal(20,18,n).clip(1,65).astype(int), 'Gender': np.random.choice([0,1],n,p=[0.6,0.4]), 'BMI': np.random.normal(15.5,2,n).clip(8,18.4), 'Weight_kg': np.random.normal(38,8,n).clip(15,55), 'Height_cm': np.random.normal(158,15,n).clip(80,185), 'Appetite_Loss': np.random.choice([0,1],n,p=[0.2,0.8]), 'Fatigue': np.random.choice([0,1],n,p=[0.15,0.85]), 'Hair_Loss': np.random.choice([0,1],n,p=[0.3,0.7]), 'Pale_Skin': np.random.choice([0,1],n,p=[0.25,0.75]), 'Result': 1})
        df_maln = pd.concat([normal_m,maln_m]).sample(frac=1,random_state=42)
        X_maln = df_maln[['Age','Gender','BMI','Weight_kg','Height_cm', 'Appetite_Loss','Fatigue','Hair_Loss','Pale_Skin']].values
        y_maln = df_maln['Result'].values
        scaler_maln = StandardScaler()
        model_maln  = MLPClassifier(hidden_layer_sizes=(32,16),activation='relu', solver='adam',max_iter=1000,random_state=42,early_stopping=True)
        model_maln.fit(scaler_maln.fit_transform(X_maln), y_maln)
        print("Malnutrition: Trained ✅")
    except: pass

    # EPIDEMIC DETECTOR
    try:
        normal_village = np.array([[8,5,3,4,3,1],[7,6,2,5,4,0],[9,5,3,4,3,1],[8,4,2,5,3,0],[7,5,3,4,4,1]])
        normal_avg = normal_village.mean(axis=0)
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(normal_village)
        print("Epidemic detector: ready ✅")
    except: pass

    models_loaded = True
    print("\nAll models loaded! AROGYA ready!")


def calculate_bmi(weight_kg, height_cm, age=25):
    height_m = height_cm / 100
    if height_m == 0: return 0, "Unknown"
    bmi = round(weight_kg / (height_m ** 2), 1)
    if bmi < 18.5: cat = "Underweight"
    elif bmi < 25: cat = "Normal"
    else: cat = "Overweight/Obese"
    return bmi, cat

def estimate_hemoglobin(spo2, pulse, age, gender):
    base = 12.0
    if spo2 < 95: base -= 1.5
    if gender == 1: base += 0.5
    return max(6.0, min(18.0, base))

def diagnose(data):
    patient_name = data.get('patient_name', 'Patient')
    
    # SAFETY NETS ADDED HERE
    age         = int(data.get('age', 0) or 0)
    gender      = int(data.get('gender', 0) or 0)
    pulse       = int(data.get('pulse', 0) or 0)
    spo2        = int(data.get('spo2', 0) or 0)
    temperature = float(data.get('temperature', 0) or 0)
    weight      = float(data.get('weight', 0) or 0)
    height      = float(data.get('height', 0) or 0)
    systolic_bp = int(data.get('systolic_bp', 0) or 0)

    headache     = bool(data.get('headache', False))
    body_ache    = bool(data.get('body_ache', False))
    rash         = bool(data.get('rash', False))
    weight_loss  = bool(data.get('weight_loss', False))
    fatigue      = bool(data.get('fatigue', False))

    diabetes_history    = int(data.get('diabetes_history', 0) or 0)
    hypertension_history= int(data.get('hypertension_history', 0) or 0)
    smoking             = int(data.get('smoking', 0) or 0)
    family_heart        = int(data.get('family_heart', 0) or 0)

    bmi, bmi_cat  = calculate_bmi(weight, height, age)
    obesity_calc  = 1 if bmi >= 27.5 else 0
    hemo_est      = estimate_hemoglobin(spo2, pulse, age, gender)
    glucose_est   = 100

    results = {}

    try:
        if model:
            d_in = np.array([[glucose_est, bmi, age, 100, systolic_bp]])
            results['Diabetes'] = round(model.predict_proba(scaler.transform(d_in))[0][1] * 100, 1)
        else: results['Diabetes'] = 0
    except: results['Diabetes'] = 0

    try:
        if model_anemia:
            a_in = np.array([[gender, hemo_est, 25, 32, 80]])
            results['Anemia'] = round(model_anemia.predict_proba(scaler_anemia.transform(a_in))[0][1] * 100, 1)
        else: results['Anemia'] = 0
    except: results['Anemia'] = 0

    try:
        if model_heart:
            h_in = np.array([[age, systolic_bp, 200, diabetes_history, hypertension_history, smoking, obesity_calc, family_heart, gender, 1]])
            results['Heart'] = round(model_heart.predict_proba(scaler_heart.transform(h_in))[0][1] * 100, 1)
        else: results['Heart'] = 0
    except: results['Heart'] = 0

    try:
        if model_ht:
            diastolic_est = systolic_bp - 40
            ht_in = np.array([[age, bmi, systolic_bp, diastolic_est, smoking, pulse]])
            results['Hypertension'] = round(model_ht.predict_proba(scaler_ht.transform(ht_in))[0][1] * 100, 1)
        else: results['Hypertension'] = 0
    except: results['Hypertension'] = 0

    try:
        if bmi < 18.5: mp = 55.0
        else: mp = 5.0
        if weight_loss: mp += 15
        results['Malnutrition'] = min(99, mp)
    except: results['Malnutrition'] = 0

    try:
        if model_dg:
            dg_in = np.array([[age, gender, temperature, pulse, spo2, 1 if headache else 0, 1 if body_ache else 0, 1 if rash else 0]])
            results['Dengue'] = round(model_dg.predict_proba(dg_in)[0][1] * 100, 1)
        else: results['Dengue'] = 0
    except: results['Dengue'] = 0

    weights = {'Diabetes':0.20,'Anemia':0.20,'Heart':0.20, 'Hypertension':0.15,'Malnutrition':0.12,'Dengue':0.13}
    score = int(sum(results[d]*weights[d] for d in results))
    risk = "HIGH" if score > 65 else "MODERATE" if score > 30 else "LOW"
    detected = [d for d,v in results.items() if v > 50]

    return {
        'patient_name': patient_name,
        'results':   results,
        'score':     score,
        'risk':      risk,
        'detected':  detected,
        'bmi':       bmi,
        'bmi_cat':   bmi_cat,
        'hemo_est':  hemo_est,
        'glucose_est': glucose_est
    }

@app.route('/')
def index(): return render_template('index.html')

@app.route('/diagnose', methods=['POST'])
def diagnose_route():
    try: return jsonify({'success': True, 'result': diagnose(request.get_json())})
    except Exception as e: return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    load_all_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
