#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import statsmodels.api as sm
import zipfile
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(BASE_DIR,"titanic_dataset", "train.csv")
TEST_PATH  = os.path.join(BASE_DIR,"titanic_dataset", "test.csv")

def main():

    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    train_data
    train_data.describe()

    sns.countplot(x="Survived",data= train_data)
    plt.title("Survival")
    plt.show()

    num_cols= []
    num_cols= train_data.select_dtypes(include=['int64','float64']).columns
    num_cols=num_cols.drop("Survived")
    num_cols=num_cols.drop("PassengerId")
    plt.figure(figsize=(20,10))
    for i,col in enumerate(num_cols,start=1):
        plt.subplot(3,3,i)
        sns.boxplot(hue="Survived",y=col,data=train_data)
        plt.title(f"{col} vs Survived")

    plt.tight_layout()
    plt.show()

    num_cols= []
    num_cols= train_data.select_dtypes(include=['int64','float64']).columns
    plt.figure(figsize=(30,20))

    for i,col in enumerate(num_cols,start=1):
        plt.subplot(3,3,i)
        sns.histplot(x=train_data[col].dropna(),data=train_data,bins=30,color="yellow",edgecolor="red")
        plt.ylabel("Frequency")
        plt.title(f"{col}'s Histogram")

    plt.tight_layout()
    plt.show()

    print(f"Null values:\n",train_data.isnull().sum())

    print(f"Duplicated data: ",train_data.duplicated().sum())

    print(f"Categorical Columns:\n",[col for col in train_data.columns if train_data[col].dtype=="object"])

    train_data['Sex'] = train_data['Sex'].astype(str).map({'male': 1, 'female': 0})
    train_data['Sex']

    test_data['Sex'] = test_data['Sex'].astype(str).map({'male': 1, 'female': 0})
    test_data['Sex']

    print(train_data['Embarked'].mode())
    train_data['Embarked'].dtype
    train_data['Embarked']= train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
    embarked_dummies = pd.get_dummies(train_data['Embarked'],prefix='Embarked',drop_first=True).astype(int)
    train_data = pd.concat([train_data,embarked_dummies], axis=1)

    test_data['Embarked']= test_data['Embarked'].fillna(test_data['Embarked'].mode()[0])
    embarked_dummies = pd.get_dummies(test_data['Embarked'],prefix='Embarked',drop_first=True).astype(int)
    test_data = pd.concat([test_data,embarked_dummies], axis=1)


    train_data["Age"] = train_data["Age"].fillna(train_data["Age"].median())
    test_data["Age"] = test_data["Age"].fillna(test_data["Age"].median()) 

    train_data["log_Fare"] = np.log1p(train_data["Fare"])
    test_data["log_Fare"] = np.log1p(test_data["Fare"])

    num_data = train_data.drop(columns=["PassengerId","Survived","Name","Ticket","Cabin","Embarked"],axis=1)

    corr_matrix = num_data.corr()

    plt.figure(figsize=(10,10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    train_data.head()
    test_data.head()

    train_data.columns

    y = train_data["Survived"]
    x_with = train_data.drop(columns=["Survived", "PassengerId", "Name","Ticket","Cabin","Embarked"],axis=1)

    x_without = train_data.drop(columns=["Parch","Fare","Survived","PassengerId","Name","Ticket","Cabin","Embarked"],axis=1)

    test_passengerId = test_data["PassengerId"]
    test_data = test_data.drop(columns=["Parch","Fare","PassengerId","Name","Ticket","Cabin","Embarked"],axis=1)
    x_with.describe()


    x_without.describe()


    x1 = sm.add_constant(x_with)
    result = sm.Logit(y, x1).fit()
    result.summary()
    print(f"AIC before reducting our dataset {result.aic}")


    x2 = sm.add_constant(x_without)
    result = sm.Logit(y, x2).fit()
    result.summary()
    print(f"AIC after reducting our dataset {result.aic}")

    X_train, X_test, y_train, y_test = train_test_split(x_without, y, test_size=0.2, random_state=7)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train,y_train)

    preds =  log_model.predict(X_test)

    print("On the training dataset, following are the results:\n")
    print(f"Accuracy: {accuracy_score(y_test, preds)}")
    print("Confusion Matrix:\n",confusion_matrix(y_test, preds))
    print("Classification Report:\n",classification_report(y_test, preds))


if __name__=="__main__":
    main()