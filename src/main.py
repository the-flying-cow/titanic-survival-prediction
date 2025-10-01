from data_load import load_data
from eda import eda_plot
from pre_process import pre_processing
from variables import variables
from model import model_predict, final_report
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
    
    preds= model_predict(X_train, y_train, X_test)
    
    final_report(y_test, preds)

if __name__=='__main__':
    main()