from flask import Flask, request, jsonify, render_template, session
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier




scaler_list = joblib.load('model\diabetis_scaler_list.pkl')
Xscaler = joblib.load('model\diabetis_Xscaler.pkl')
model=joblib.load('model\DiabetisRisk.pkl')

def preprocess_test(xtest, scaler_list):
    for i,scaler in enumerate(scaler_list):
        xtest[:,i]=scaler.transform(xtest[:,i].reshape(-1,1)).reshape(-1,)
    return xtest
decode={1:"The patient's risk of diabetes is elevated. It is advisable to conduct a confirmatory test to ensure accurate diagnosis and appropriate management.",
        0:"The patient's risk of diabetes is not elevated. A confirmatory test may not be necessary at this time."}

app = Flask(__name__)
app.secret_key =os.urandom(24)

@app.route('/') 
def dia_home():
    return render_template('dia_home.html')



@app.route('/predict', methods=['POST'])
def predict():
    patientName = request.form['patientName'].strip().title() if request.form['patientName'].strip() else "NOT SUPPLIED"
    clinicCardNumber = request.form['clinicCardNumber'].strip().upper() if request.form['clinicCardNumber'].strip() else "NOT SUPPLIED"
    pregnancy=request.form['pregnancy']
    glucose=request.form['glucose']
    BloodPressure=request.form['BloodPressure']
    SkinThickness=request.form['SkinThickness']
    Insulin=request.form['Insulin']
    BMI=request.form['BMI']
    DiabetesPedigreeFunction=request.form['DiabetesPedigreeFunction']
    Age=request.form['Age']

    info={'Pregnancy':pregnancy,
       'Glucose_level':glucose,	
       'BloodPressure':BloodPressure,
       'SkinThickness':SkinThickness,	
       'Insulin':Insulin,	
       'BMI':BMI,
       'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
       'Age':Age,
      }
    
    info=pd.DataFrame(info, index=[0])
    predictor=info.to_numpy()
    predictor=preprocess_test(predictor, scaler_list)
    predictor=Xscaler.transform(predictor)

    prediction=model.predict(predictor)

    return render_template('dia_result.html', 
                           pregnancy=pregnancy, 
                           glucose=glucose, 
                           BloodPressure=BloodPressure, 
                           SkinThickness=SkinThickness, 
                           Insulin=Insulin, BMI=BMI, 
                           DiabetesPedigreeFunction=DiabetesPedigreeFunction, 
                           Age=Age, patientName = patientName,
                           clinicCardNumber = clinicCardNumber,
                           prediction=prediction[0])
    



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)

