from data_load import load_data
from pre_process import pre_processing
from variables import variables
from model import hyper_parameter_tune
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import joblib
from sklearn import set_config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR,"titanic_dataset", "train.csv")

def main():
    df= load_data(TRAIN_PATH)

    df= pre_processing(df)
    
    y = df["Survived"]
    data= variables(df)
    X_train, X_test, y_train, y_test = train_test_split(data, y,test_size=0.2, random_state=7)
    
    log_pipe= Pipeline([("standard_scaling", StandardScaler()), ("classifier", LogisticRegression(max_iter= 1000))])

    rf_pipe= Pipeline([("standard_scaling", StandardScaler()), ("classifier", RandomForestClassifier(random_state= 7))])
    rf_pipe= hyper_parameter_tune(rf_pipe, X_train, y_train)

    models_dir= os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok= True)
    model_path= os.path.join(models_dir, "best_model.pkl")
    joblib.dump(rf_pipe, model_path)
    
    return rf_pipe

if __name__=='__main__':
    main()