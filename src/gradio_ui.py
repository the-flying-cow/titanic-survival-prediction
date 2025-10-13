from main import main
from model import predict_survival
import gradio as gr
import numpy as np
import joblib
import os

BASE_DIR= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH= os.path.join(BASE_DIR, "models", "best_model.pkl")
if os.path.exists(MODEL_PATH):
    pipe= joblib.load(MODEL_PATH)
else:
    pipe= main()
def predict_inputs(pclass, sex, age, sibsp, embarked, fare):
    try:
        pclass = int(pclass)
        sex = str(sex)
        age = int(age)
        sibsp = int(sibsp)
        embarked = str(embarked)
        fare = int(fare)
    except:
        raise ValueError("Please give correct inputs")

    log_fare= np.log(fare + 1)
    
    if embarked == "Q":
        embarked_q, embarked_s = 1, 0
    elif embarked == "S":
        embarked_q, embarked_s = 0, 1
    else:
        embarked_q, embarked_s = 0, 0
    
    if sex == 'male':
        sex= 1
    else:
        sex= 0
    
    features= [[pclass, sex, age, sibsp, embarked_q, embarked_s, log_fare]]
    return "Survived" if predict_survival(pipe, features)[0] == 1 else "Not Survived"

inputs= [
        gr.Dropdown([1,2,3], label="Pclass"),
        gr.Radio(["male", "female"], label="Sex"),
        gr.Number(label="Age"),
        gr.Number(label="Siblings/Spouses aboard", value=0),
        gr.Radio(["Q", "S", "C"], label= "Embarked"),
        gr.Number(label="Fare")
    ]

gr.Interface(fn= predict_inputs, inputs= inputs, outputs= "text", 
            title= 'Titanic Survival Prediction',
            description='Current accuracy is only around 78%.. will be improving that soon. Stay Tuned😉').launch(share=True)
