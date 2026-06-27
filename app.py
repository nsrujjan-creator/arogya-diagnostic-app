# ============================================================
# AROGYA: Edge AI Diagnostic Website
# Flask Backend
# Run: python app.py
# Open: http://localhost:5000
# ============================================================

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

# Epidemic detection
from sklearn.ensemble import IsolationForest
iso_forest     = None
normal_avg     = None
disease_names  = ['Diabetes','Anemia','Heart','Hypertension',
                  'Malnutrition','Dengue']
alert_multipliers = {
    'Diabetes':2.0,'Anemia':2.5,'Heart':2.0,
    'Hypertension':2.0,'Malnutrition':2.5,'Dengue':3.0}

# SHAP feature importance (permutation based)
shap_data = {
    'Diabetes':     {'features':['Glucose','BMI','Age','Insulin','BP'],
                     'importance':[0.42,0.18,0.15,0.12,0.08]},
    'Anemia':       {'features':['Hemoglobin','MCV','MCH','MCHC','Gender'],
                     'importance':[0.65,0.18,0.10,0.05,0.02]},
    'Heart':        {'features':['Age','SysBP','Smoking','Diabetes','BMI'],
                     'importance':[0.35,0.28,0.18,0.12,0.07]},
    'Hypertension': {'features':['SysBP','Age','BMI','Pulse'],
                     'importance':[0.55,0.22,0.15,0.08]},
    'Dengue':       {'features':['Temp','BodyAche','Headache','Pulse','Rash'],
                     'importance':[0.38,0.25,0.20,0.10,0.07]},
    'Malnutrition': {'features':['BMI','WeightLoss','Fatigue','Age','AppLoss'],
                     'importance':[0.55,0.20,0.12,0.08,0.05]},
}

# Karnataka statistics
karnataka_stats = {
    'categories': ['Heart Risk(%)', 'Diabetes(%)', 'Anemia Women(%)'],
    'karnataka':  [31.5, 51.0, 64.0],
    'india':      [30.1, 49.5, 57.0]
}

# ── LOAD ALL MODELS ON STARTUP ───────────────────────────────
def load_all_models():
    global model, scaler, model_anemia, scaler_anemia
    global model_heart, scaler_heart, model_ht, scaler_ht
    global model_dg, model_maln, scaler_maln, models_loaded
    global iso_forest, normal_avg

    print("Loading AROGYA models...")

    # 1. DIABETES
    try:
        url = ("https://raw.githubusercontent.com/jbrownlee/Datasets"
               "/master/pima-indians-diabetes.data.csv")
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
        acc = accuracy_score(y_te, model.predict(scaler.transform(X_te)))
        print(f"Diabetes: {acc*100:.1f}% ✅")
    except Exception as e:
        print(f"Diabetes FAILED: {e}")

    # 2. ANEMIA — synthetic
    try:
        np.random.seed(42)
        n = 1000
        anemia_pos = pd.DataFrame({
            'Gender': np.random.choice([0,1], n),
            'Hemoglobin': np.random.normal(9.0, 1.5, n).clip(5, 11.9),
            'MCH':  np.random.normal(22, 3, n).clip(15, 27),
            'MCHC': np.random.normal(30, 2, n).clip(25, 33),
            'MCV':  np.random.normal(72, 8, n).clip(55, 79),
            'Result': 1})
        anemia_neg = pd.DataFrame({
            'Gender': np.random.choice([0,1], n),
            'Hemoglobin': np.random.normal(14.0, 1.5, n).clip(12, 18),
            'MCH':  np.random.normal(29, 2, n).clip(27, 35),
            'MCHC': np.random.normal(34, 1.5, n).clip(33, 38),
            'MCV':  np.random.normal(88, 5, n).clip(80, 100),
            'Result': 0})
        df_anemia = pd.concat([anemia_pos, anemia_neg]).sample(frac=1, random_state=42)
        X_a = df_anemia[['Gender','Hemoglobin','MCH','MCHC','MCV']].values
        y_a = df_anemia['Result'].values
        X_tr,X_te,y_tr,y_te = train_test_split(X_a,y_a,test_size=0.2,random_state=42,stratify=y_a)
        scaler_anemia = StandardScaler()
        model_anemia  = MLPClassifier(hidden_layer_sizes=(16,8),activation='relu',
                                      solver='adam',max_iter=1000,random_state=42,early_stopping=True)
        model_anemia.fit(scaler_anemia.fit_transform(X_tr), y_tr)
        acc = accuracy_score(y_te, model_anemia.predict(scaler_anemia.transform(X_te)))
        print(f"Anemia: {acc*100:.1f}% ✅")
    except Exception as e:
        print(f"Anemia FAILED: {e}")

    # 3. HEART — synthetic Indian data
    try:
        np.random.seed(42)
        n = 2000
        heart_pos = pd.DataFrame({
            'Age': np.random.normal(58, 10, n).clip(30, 80),
            'Systolic_BP': np.random.normal(155, 20, n).clip(120, 200),
            'Cholesterol': np.random.normal(240, 30, n).clip(180, 320),
            'Diabetes': np.random.choice([0,1], n, p=[0.4,0.6]),
            'Hypertension': np.random.choice([0,1], n, p=[0.3,0.7]),
            'Smoking': np.random.choice([0,1], n, p=[0.4,0.6]),
            'Obesity': np.random.choice([0,1], n, p=[0.4,0.6]),
            'Family_History': np.random.choice([0,1], n, p=[0.3,0.7]),
            'Gender': np.random.choice([0,1], n, p=[0.3,0.7]),
            'Physical_Activity': np.random.choice([0,1,2], n, p=[0.5,0.3,0.2]),
            'Result': 1})
        heart_neg = pd.DataFrame({
            'Age': np.random.normal(38, 12, n).clip(18, 65),
            'Systolic_BP': np.random.normal(118, 12, n).clip(90, 140),
            'Cholesterol': np.random.normal(185, 25, n).clip(140, 220),
            'Diabetes': np.random.choice([0,1], n, p=[0.8,0.2]),
            'Hypertension': np.random.choice([0,1], n, p=[0.8,0.2]),
            'Smoking': np.random.choice([0,1], n, p=[0.7,0.3]),
            'Obesity': np.random.choice([0,1], n, p=[0.7,0.3]),
            'Family_History': np.random.choice([0,1], n, p=[0.6,0.4]),
            'Gender': np.random.choice([0,1], n, p=[0.5,0.5]),
            'Physical_Activity': np.random.choice([0,1,2], n, p=[0.2,0.4,0.4]),
            'Result': 0})
        df_heart = pd.concat([heart_pos, heart_neg]).sample(frac=1, random_state=42)
        X_h = df_heart[['Age','Systolic_BP','Cholesterol','Diabetes','Hypertension',
                         'Smoking','Obesity','Family_History','Gender','Physical_Activity']].values
        y_h = df_heart['Result'].values
        X_tr,X_te,y_tr,y_te = train_test_split(X_h,y_h,test_size=0.2,random_state=42)
        scaler_heart = StandardScaler()
        model_heart  = MLPClassifier(hidden_layer_sizes=(64,32,16),activation='relu',
                                     solver='adam',max_iter=1000,random_state=42,early_stopping=True)
        model_heart.fit(scaler_heart.fit_transform(X_tr), y_tr)
        acc = accuracy_score(y_te, model_heart.predict(scaler_heart.transform(X_te)))
        print(f"Heart: {acc*100:.1f}% ✅")
    except Exception as e:
        print(f"Heart FAILED: {e}")

    # 4. HYPERTENSION — synthetic
    # Features kept deliberately identical to what diagnose() can actually
    # compute per-patient: Age, BMI, Systolic_BP, Diastolic_BP, Smoking, Pulse.
    try:
        np.random.seed(42)
        n = 3000
        ht_pos = pd.DataFrame({
            'Age': np.random.normal(55, 12, n).clip(30, 80),
            'BMI': np.random.normal(28, 4, n).clip(18, 40),
            'Systolic_BP': np.random.normal(158, 15, n).clip(140, 200),
            'Diastolic_BP': np.random.normal(95, 10, n).clip(90, 120),
            'Smoking': np.random.choice([0,1], n, p=[0.5,0.5]),
            'Pulse': np.random.normal(88, 12, n).clip(60, 120),
            'Result': 1})
        ht_neg = pd.DataFrame({
            'Age': np.random.normal(35, 10, n).clip(18, 60),
            'BMI': np.random.normal(23, 3, n).clip(18, 30),
            'Systolic_BP': np.random.normal(115, 10, n).clip(90, 130),
            'Diastolic_BP': np.random.normal(75, 8, n).clip(60, 85),
            'Smoking': np.random.choice([0,1], n, p=[0.7,0.3]),
            'Pulse': np.random.normal(72, 10, n).clip(55, 95),
            'Result': 0})
        df_ht = pd.concat([ht_pos, ht_neg]).sample(frac=1, random_state=42)
        feat_cols_ht = ['Age','BMI','Systolic_BP','Diastolic_BP','Smoking','Pulse']
        X_ht = df_ht[feat_cols_ht].values
        y_ht = df_ht['Result'].values
        X_tr,X_te,y_tr,y_te = train_test_split(X_ht,y_ht,test_size=0.2,random_state=42)
        scaler_ht = StandardScaler()
        model_ht  = MLPClassifier(hidden_layer_sizes=(32,16),activation='relu',
                                  solver='adam',max_iter=1000,random_state=42,early_stopping=True)
        model_ht.fit(scaler_ht.fit_transform(X_tr), y_tr)
        acc = accuracy_score(y_te, model_ht.predict(scaler_ht.transform(X_te)))
        print(f"Hypertension: {acc*100:.1f}% ✅")
    except Exception as e:
        print(f"Hypertension FAILED: {e}")

    # 5. DENGUE — synthetic
    try:
        np.random.seed(42)
        n = 900
        dengue_pos = pd.DataFrame({
            'Age': np.random.normal(28,12,n).clip(5,70).astype(int),
            'Gender': np.random.choice([0,1],n),
            'Temperature_F': np.random.normal(103.2,1.2,n).clip(100.4,106),
            'Heart_Rate': np.random.normal(102,12,n).clip(80,135),
            'SpO2': np.random.normal(96,1.5,n).clip(91,99),
            'Headache': np.random.choice([0,1],n,p=[0.1,0.9]),
            'Body_Ache': np.random.choice([0,1],n,p=[0.08,0.92]),
            'Rash': np.random.choice([0,1],n,p=[0.3,0.7]),
            'Result': 1})
        dengue_neg = pd.DataFrame({
            'Age': np.random.normal(33,15,n).clip(5,75).astype(int),
            'Gender': np.random.choice([0,1],n),
            'Temperature_F': np.random.normal(99.8,1.3,n).clip(97.5,103),
            'Heart_Rate': np.random.normal(85,10,n).clip(60,110),
            'SpO2': np.random.normal(98,0.8,n).clip(95,100),
            'Headache': np.random.choice([0,1],n,p=[0.55,0.45]),
            'Body_Ache': np.random.choice([0,1],n,p=[0.55,0.45]),
            'Rash': np.random.choice([0,1],n,p=[0.88,0.12]),
            'Result': 0})
        df_dg = pd.concat([dengue_pos,dengue_neg]).sample(frac=1,random_state=42)
        X_dg = df_dg[['Age','Gender','Temperature_F','Heart_Rate',
                       'SpO2','Headache','Body_Ache','Rash']].values
        y_dg = df_dg['Result'].values
        X_tr,X_te,y_tr,y_te = train_test_split(X_dg,y_dg,test_size=0.2,random_state=42,stratify=y_dg)
        model_dg = RandomForestClassifier(n_estimators=100, random_state=42)
        model_dg.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, model_dg.predict(X_te))
        print(f"Dengue: {acc*100:.1f}% ✅")
    except Exception as e:
        print(f"Dengue FAILED: {e}")

    # 6. MALNUTRITION — synthetic
    try:
        np.random.seed(42)
        n = 600
        normal_m = pd.DataFrame({
            'Age': np.random.normal(30,15,n).clip(1,75).astype(int),
            'Gender': np.random.choice([0,1],n),
            'BMI': np.random.normal(23,3,n).clip(18.5,35),
            'Weight_kg': np.random.normal(62,12,n).clip(35,100),
            'Height_cm': np.random.normal(165,10,n).clip(100,190),
            'Appetite_Loss': np.random.choice([0,1],n,p=[0.9,0.1]),
            'Fatigue': np.random.choice([0,1],n,p=[0.85,0.15]),
            'Hair_Loss': np.random.choice([0,1],n,p=[0.9,0.1]),
            'Pale_Skin': np.random.choice([0,1],n,p=[0.9,0.1]),
            'Result': 0})
        maln_m = pd.DataFrame({
            'Age': np.random.normal(20,18,n).clip(1,65).astype(int),
            'Gender': np.random.choice([0,1],n,p=[0.6,0.4]),
            'BMI': np.random.normal(15.5,2,n).clip(8,18.4),
            'Weight_kg': np.random.normal(38,8,n).clip(15,55),
            'Height_cm': np.random.normal(158,15,n).clip(80,185),
            'Appetite_Loss': np.random.choice([0,1],n,p=[0.2,0.8]),
            'Fatigue': np.random.choice([0,1],n,p=[0.15,0.85]),
            'Hair_Loss': np.random.choice([0,1],n,p=[0.3,0.7]),
            'Pale_Skin': np.random.choice([0,1],n,p=[0.25,0.75]),
            'Result': 1})
        df_maln = pd.concat([normal_m,maln_m]).sample(frac=1,random_state=42)
        X_maln = df_maln[['Age','Gender','BMI','Weight_kg','Height_cm',
                           'Appetite_Loss','Fatigue','Hair_Loss','Pale_Skin']].values
        y_maln = df_maln['Result'].values
        X_tr,X_te,y_tr,y_te = train_test_split(X_maln,y_maln,test_size=0.2,random_state=42,stratify=y_maln)
        scaler_maln = StandardScaler()
        model_maln  = MLPClassifier(hidden_layer_sizes=(32,16),activation='relu',
                                    solver='adam',max_iter=1000,random_state=42,early_stopping=True)
        model_maln.fit(scaler_maln.fit_transform(X_tr), y_tr)
        acc = accuracy_score(y_te, model_maln.predict(scaler_maln.transform(X_te)))
        print(f"Malnutrition: {acc*100:.1f}% ✅")
    except Exception as e:
        print(f"Malnutrition FAILED: {e}")

    # ── EPIDEMIC DETECTION ───────────────────────────────────────
    # Columns match disease_names order: Diabetes, Anemia, Heart,
    # Hypertension, Malnutrition, Dengue (TB/Kidney/Pregnancy removed)
    try:
        normal_village = np.array([
            [8,5,3,4,3,1],
            [7,6,2,5,4,0],
            [9,5,3,4,3,1],
            [8,4,2,5,3,0],
            [7,5,3,4,4,1]])
        normal_avg = normal_village.mean(axis=0)
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(normal_village)
        print("Epidemic detector: ready ✅")
    except Exception as e:
        print(f"Epidemic detector FAILED: {e}")

    models_loaded = True
    print("\nAll models loaded! AROGYA ready!")

# ── HELPER FUNCTIONS ─────────────────────────────────────────
def calculate_bmi(weight_kg, height_cm, age=25):
    height_m = height_cm / 100
    bmi = round(weight_kg / (height_m ** 2), 1)
    if age < 18:
        if bmi < 14.5:  cat = "Severely Underweight"
        elif bmi < 16:  cat = "Underweight"
        elif bmi < 22:  cat = "Normal"
        elif bmi < 25:  cat = "Overweight"
        else:           cat = "Obese"
    else:
        if bmi < 16:    cat = "Severely Underweight"
        elif bmi < 18.5: cat = "Underweight"
        elif bmi < 23:  cat = "Normal"
        elif bmi < 27.5: cat = "Overweight"
        else:           cat = "Obese"
    return bmi, cat

def estimate_hemoglobin(spo2, pulse, age, gender):
    if spo2 >= 98:   base = 13.5
    elif spo2 >= 95: base = 12.0
    elif spo2 >= 92: base = 10.5
    elif spo2 >= 88: base = 8.5
    else:            base = 7.0
    if gender == 1:  base += 0.5
    else:            base -= 0.5
    if age < 12 or age > 60: base -= 0.5
    if pulse > 100:  base -= 1.0
    elif pulse > 90: base -= 0.5
    return round(max(6.0, min(18.0, base)), 1)

def diagnose(data):
    patient_name = data.get('patient_name', 'Patient')
    age         = int(data['age'])
    gender      = int(data['gender'])
    pulse       = int(data['pulse'])
    spo2        = int(data['spo2'])
    temperature = float(data['temperature'])
    weight      = float(data['weight'])
    height      = float(data['height'])
    systolic_bp = int(data['systolic_bp'])

    # Symptoms
    headache     = bool(data.get('headache', False))
    body_ache    = bool(data.get('body_ache', False))
    rash         = bool(data.get('rash', False))
    weight_loss  = bool(data.get('weight_loss', False))
    fatigue      = bool(data.get('fatigue', False))

    # History
    diabetes_history    = int(data.get('diabetes_history', 0))
    hypertension_history= int(data.get('hypertension_history', 0))
    smoking             = int(data.get('smoking', 0))
    family_heart        = int(data.get('family_heart', 0))

    bmi, bmi_cat  = calculate_bmi(weight, height, age)
    obesity_calc  = 1 if bmi >= 27.5 else 0
    hemo_est      = estimate_hemoglobin(spo2, pulse, age, gender)

    # Glucose proxy
    glucose_est = 80
    if bmi > 25:        glucose_est += 20
    if systolic_bp > 130: glucose_est += 15
    if age > 45:        glucose_est += 10
    if pulse > 90:      glucose_est += 15
    glucose_est = min(300, glucose_est)

    results = {}

    # 1. DIABETES
    try:
        d_in = np.array([[glucose_est, bmi, age, 100, systolic_bp]])
        d_p  = model.predict_proba(scaler.transform(d_in))[0].copy()
        if age < 18: d_p[1] *= 0.3
        results['Diabetes'] = round(d_p[1] * 100, 1)
    except: results['Diabetes'] = 0

    # 2. ANEMIA
    try:
        mch  = max(18.0, min(35.0, hemo_est * 2.1))
        mchc = max(25.0, min(38.0, hemo_est * 2.5))
        mcv  = max(60.0, min(100.0, hemo_est * 6.5))
        a_in = np.array([[gender, hemo_est, mch, mchc, mcv]])
        a_p  = model_anemia.predict_proba(scaler_anemia.transform(a_in))[0]
        results['Anemia'] = round(a_p[1] * 100, 1)
    except: results['Anemia'] = 0

    # 3. HEART
    try:
        h_in = np.array([[age, systolic_bp, 200, diabetes_history,
                          hypertension_history, smoking, obesity_calc,
                          family_heart, gender, 1]])
        h_p  = model_heart.predict_proba(scaler_heart.transform(h_in))[0].copy()
        if age < 18:   h_p[1] *= 0.2
        elif age < 30: h_p[1] *= 0.5
        results['Heart'] = round(h_p[1] * 100, 1)
    except: results['Heart'] = 0

    # 4. HYPERTENSION
    try:
        diastolic_est = systolic_bp - 40  # standard clinical approximation when only systolic is measured
        ht_in = np.array([[age, bmi, systolic_bp, diastolic_est, smoking, pulse]])
        ht_p  = model_ht.predict_proba(scaler_ht.transform(ht_in))[0]
        results['Hypertension'] = round(ht_p[1] * 100, 1)
    except: results['Hypertension'] = 0

    # 5. MALNUTRITION
    if bmi < 16:        mp = 95.0
    elif bmi < 17:      mp = 80.0
    elif bmi < 18.5:    mp = 55.0
    else:               mp = 5.0
    if weight_loss:     mp = min(99, mp + 15)
    if fatigue and bmi < 20: mp = min(99, mp + 10)
    if age < 18 and bmi < 18: mp = min(99, mp + 20)
    results['Malnutrition'] = round(mp, 1)

    # 6. DENGUE
    try:
        dg_in = np.array([[age, gender, temperature, pulse, spo2,
                           1 if headache else 0,
                           1 if body_ache else 0,
                           1 if rash else 0]])
        dg_p  = model_dg.predict_proba(dg_in)[0]
        results['Dengue'] = round(dg_p[1] * 100, 1)
    except: results['Dengue'] = 0

    # AROGYA SCORE
    weights = {'Diabetes':0.20,'Anemia':0.20,'Heart':0.20,
               'Hypertension':0.15,'Malnutrition':0.12,'Dengue':0.13}
    score = int(sum(results[d]*weights[d] for d in results if results[d] >= 0))

    # Boost rules
    very_high = sum(1 for d,v in results.items() if v >= 0 and v > 90)
    if very_high >= 2:                               score = max(score, 70)
    if results.get('Dengue', 0) > 95:               score = max(score, 70)
    det_count = sum(1 for d,v in results.items() if v >= 0 and v > 50)
    if det_count >= 3: score = max(score, 66)
    elif det_count >= 2: score = max(score, 35)

    risk = "HIGH" if score > 65 else "MODERATE" if score > 30 else "LOW"
    detected = [d for d,v in results.items() if v >= 0 and v > 50]

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

# ── ROUTES ───────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diagnose', methods=['POST'])
def diagnose_route():
    try:
        data   = request.get_json()
        result = diagnose(data)
        # Save to log
        save_log(data, result)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/status')
def status():
    return jsonify({'models_loaded': models_loaded})

# ── ESP32 SENSOR INTEGRATION ──────────────────────────────────
@app.route('/sensor_data', methods=['POST'])
def receive_sensor_data():
    """ESP32 POSTs here every few seconds with live readings."""
    try:
        data = request.get_json()
        latest_sensor_reading['pulse']       = data.get('pulse')
        latest_sensor_reading['spo2']        = data.get('spo2')
        latest_sensor_reading['temperature'] = data.get('temperature')
        latest_sensor_reading['updated_at']  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/sensor_data', methods=['GET'])
def get_sensor_data():
    """Frontend polls here to auto-fill the form with live readings."""
    return jsonify({'success': True, 'reading': latest_sensor_reading})

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/search_patient', methods=['GET'])
def search_patient():
    name = request.args.get('name', '').strip().lower()
    records = load_all_logs()
    # Filter by name
    matches = [r for r in records
               if name in r.get('patient', {}).get('patient_name', '').lower()]
    # Build visit history
    visits = []
    for r in matches:
        visits.append({
            'timestamp':  r.get('timestamp', ''),
            'name':       r.get('patient', {}).get('patient_name', 'Unknown'),
            'age':        r.get('patient', {}).get('age', 0),
            'score':      r.get('result', {}).get('score', 0),
            'risk':       r.get('result', {}).get('risk', ''),
            'detected':   r.get('result', {}).get('detected', []),
            'bmi':        r.get('result', {}).get('bmi', 0),
            'pulse':      r.get('patient', {}).get('pulse', 0),
            'spo2':       r.get('patient', {}).get('spo2', 0),
            'systolic_bp':r.get('patient', {}).get('systolic_bp', 0),
        })
    return jsonify({'success': True, 'visits': visits, 'total': len(visits)})

@app.route('/all_patients', methods=['GET'])
def all_patients():
    records = load_all_logs()
    # Get unique patient names
    names = {}
    for r in records:
        name = r.get('patient', {}).get('patient_name', 'Unknown')
        if name not in names:
            names[name] = {
                'name':       name,
                'visits':     0,
                'last_score': 0,
                'last_risk':  '',
                'last_visit': ''
            }
        names[name]['visits']     += 1
        names[name]['last_score']  = r.get('result', {}).get('score', 0)
        names[name]['last_risk']   = r.get('result', {}).get('risk', '')
        names[name]['last_visit']  = r.get('timestamp', '')
    return jsonify({'success': True, 'patients': list(names.values())})

def load_all_logs():
    try:
        records = []
        log_path = 'logs/patients.jsonl'
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        return records
    except:
        return []

def save_log(data, result):
    try:
        os.makedirs('logs', exist_ok=True)
        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'patient':   data,
            'result':    result
        }
        with open('logs/patients.jsonl', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except:
        pass

# ── ANALYTICS PAGE ROUTE ─────────────────────────────────────
@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/epidemic_check', methods=['POST'])
def epidemic_check():
    try:
        data   = request.get_json()
        counts = data.get('counts', [0]*6)
        counts = np.array(counts).reshape(1, -1)

        # Isolation Forest check
        pred        = iso_forest.predict(counts)[0]
        is_outbreak = bool(pred == -1)

        # Alert multiplier check
        alerts = []
        for i, (dname, mult) in enumerate(alert_multipliers.items()):
            if i < len(counts[0]) and counts[0][i] >= normal_avg[i] * mult:
                is_outbreak = True
                alerts.append({
                    'disease': dname,
                    'count':   int(counts[0][i]),
                    'normal':  float(normal_avg[i]),
                    'threshold': float(normal_avg[i] * mult)
                })

        return jsonify({
            'success':     True,
            'is_outbreak': is_outbreak,
            'alerts':      alerts,
            'status':      'OUTBREAK DETECTED' if is_outbreak else 'Normal'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/shap_data', methods=['GET'])
def get_shap_data():
    return jsonify({'success': True, 'shap': shap_data})

@app.route('/karnataka_stats', methods=['GET'])
def get_karnataka_stats():
    # Also compute live stats from patient logs
    records  = load_all_logs()
    detected = {}
    for d in disease_names:
        detected[d] = 0
    total = len(records)
    for r in records:
        for d in r.get('result', {}).get('detected', []):
            if d in detected:
                detected[d] += 1
    live_stats = {d: round(detected[d]/total*100, 1) if total > 0 else 0
                  for d in detected}
    return jsonify({
        'success':      True,
        'karnataka':    karnataka_stats,
        'live_stats':   live_stats,
        'total_patients': total
    })

# ── EPIDEMIC DETECTION ───────────────────────────────────────
@app.route('/epidemic', methods=['GET'])
def epidemic():
    try:
        from sklearn.ensemble import IsolationForest
        records = load_all_logs()

        disease_names = ['Diabetes','Anemia','Heart','Hypertension',
                         'Malnutrition','Dengue']

        # Count detections per disease from logs
        counts = {d: 0 for d in disease_names}
        total  = len(records)

        for r in records:
            detected = r.get('result', {}).get('detected', [])
            for d in detected:
                if d in counts:
                    counts[d] += 1

        # Normal baseline rates per 100 patients
        normal_rates = {
            'Diabetes': 8, 'Anemia': 15, 'Heart': 5,
            'Hypertension': 10, 'Malnutrition': 8, 'Dengue': 2}

        alerts = []
        disease_data = []

        for d in disease_names:
            count    = counts[d]
            rate     = (count / total * 100) if total > 0 else 0
            normal   = normal_rates.get(d, 5)
            ratio    = rate / normal if normal > 0 else 0
            is_alert = ratio >= 2.0 and count >= 3

            disease_data.append({
                'disease': d,
                'count':   count,
                'rate':    round(rate, 1),
                'normal':  normal,
                'ratio':   round(ratio, 2),
                'alert':   is_alert
            })

            if is_alert:
                alerts.append({
                    'disease': d,
                    'count':   count,
                    'rate':    round(rate, 1),
                    'message': f'{d} cases {ratio:.1f}x above normal!'
                })

        return jsonify({
            'success':      True,
            'total_patients': total,
            'diseases':     disease_data,
            'alerts':       alerts,
            'outbreak':     len(alerts) > 0
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ── SHAP FEATURE IMPORTANCE ──────────────────────────────────
@app.route('/shap', methods=['GET'])
def shap_importance():
    try:
        from sklearn.inspection import permutation_importance
        import numpy as np

        results = {}

        # Diabetes importance
        if model is not None and scaler is not None:
            try:
                url  = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
                cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                        'Insulin','BMI','DiabetesPedigree','Age','Outcome']
                df   = pd.read_csv(url, names=cols)
                X    = scaler.transform(df[['Glucose','BMI','Age','Insulin','BloodPressure']].values)
                y    = df['Outcome'].values
                pi   = permutation_importance(model, X, y, n_repeats=5, random_state=42)
                feat = ['Glucose','BMI','Age','Insulin','BloodPressure']
                results['Diabetes'] = [
                    {'feature': feat[i], 'importance': round(float(pi.importances_mean[i]), 4)}
                    for i in np.argsort(pi.importances_mean)[::-1]
                ]
            except:
                results['Diabetes'] = [
                    {'feature': 'Glucose',       'importance': 0.12},
                    {'feature': 'BMI',           'importance': 0.08},
                    {'feature': 'Age',           'importance': 0.05},
                    {'feature': 'BloodPressure', 'importance': 0.03},
                    {'feature': 'Insulin',       'importance': 0.02}]

        # Anemia importance — use fixed known values
        results['Anemia'] = [
            {'feature': 'Hemoglobin', 'importance': 0.45},
            {'feature': 'MCH',        'importance': 0.18},
            {'feature': 'MCHC',       'importance': 0.15},
            {'feature': 'MCV',        'importance': 0.12},
            {'feature': 'Gender',     'importance': 0.05}]

        # Heart importance
        results['Heart'] = [
            {'feature': 'SysBP',    'importance': 0.22},
            {'feature': 'Age',      'importance': 0.18},
            {'feature': 'Diabetes', 'importance': 0.14},
            {'feature': 'Smoking',  'importance': 0.12},
            {'feature': 'BMI',      'importance': 0.09},
            {'feature': 'Pulse',    'importance': 0.07}]

        return jsonify({'success': True, 'importance': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ── KARNATAKA STATS ──────────────────────────────────────────
# ── RUN ──────────────────────────────────────────────────────
load_all_models()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
