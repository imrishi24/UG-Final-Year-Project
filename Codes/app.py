from flask import Flask, render_template, request
from csv import DictWriter
import requests
import joblib
import os
import  numpy as np
import pandas as pd
import pickle
from flask_ngrok import run_with_ngrok



app= Flask(__name__)
run_with_ngrok(app)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/result/",methods=['POST','GET'])
def result():
    if  request.method=="GET":
        return render_template("ritu.html")
    elif request.method=="POST":
        gender=int(request.form['gender'])
        age=int(request.form['age'])
        hypertension=int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = int(request.form['ever_married'])
        work_type = int(request.form['work_type'])
        Residence_type = int(request.form['Residence_type'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = int(request.form['smoking_status'])
       
        x=np.array([gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,
                avg_glucose_level,bmi,smoking_status]).reshape(1,-1)

        field_names = ['gender','age','hypertension',
               'heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status']

        dict={'gender':gender,'age':age,'hypertension':hypertension,
               'heart_disease':heart_disease,'ever_married':ever_married,'work_type':work_type,'Residence_type':Residence_type,'avg_glucose_level':avg_glucose_level,'bmi':bmi,'smoking_status':smoking_status}
       
        with open('demo.csv', 'a') as f_object:
            
            
            dictwriter_object = DictWriter(f_object, fieldnames=field_names)
        
            
            dictwriter_object.writerow(dict)
        
            f_object.close()

     
    scaler_path=os.path.join('F:/rishabh/8th sem/MAJOR PROJECT/Codes/models/scaler.pkl')
    scaler=None
    with open(scaler_path,'rb') as scaler_file:
        scaler=pickle.load(scaler_file)

    x=scaler.transform(x)

    model_path=os.path.join('F:/rishabh/8th sem/MAJOR PROJECT/Codes/models/rf.sav')
    rf=joblib.load(model_path)

    Y_pred=rf.predict(x)
    Y_pred_prob = rf.predict_proba(x)
    a = Y_pred_prob[0][1]*100
    r=int(a)

    # for No Stroke Risk
    if Y_pred==0 and r<=50:
        return render_template('nostroke.html', value=r)
    else:
        return render_template('stroke.html', value=r)

@app.route("/about/")
def about():
    return render_template("about.html")

if __name__=="__main__":
    app.run()