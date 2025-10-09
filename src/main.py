from data_load import load_data
from eda import eda_plot
from pre_process import pre_processing
from variables import variables
from model import model_predict, hyper_parameter_tune, final_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
from sklearn import set_config


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR,"titanic_dataset", "train.csv")

def main():
    df= load_data(TRAIN_PATH)
    
    eda_plot(df)
    
    df= pre_processing(df)
    
    y = df["Survived"]
    data= variables(df)
    X_train, X_test, y_train, y_test = train_test_split(data, y,test_size=0.2, random_state=7)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    

    log_pipe= Pipeline([("standard_scaling", StandardScaler()), ("classifier", LogisticRegression(max_iter= 1000))])
    set_config(display= 'diagram')
    print(log_pipe)
    log_preds= model_predict(log_pipe,X_train, y_train, X_test)
    final_report(y_test, log_preds)

    rf_pipe= Pipeline([("standard_scaling", StandardScaler()), ("classifier", RandomForestClassifier(random_state= 7))])
    rf_pipe= hyper_parameter_tune(rf_pipe, X_train, y_train)
    set_config(display= 'diagram')
    print(rf_pipe)
    rf_preds= model_predict(rf_pipe,X_train, y_train, X_test)
    final_report(y_test, rf_preds)

if __name__=='__main__':
    main()