#!/usr/bin/env python
# coding: utf-8

# First we import all the necessary libraries that we will need.

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
    
    # Upon extracting our zipfile we can see it contains- 
    # test.csv, train.csv
    # We will be training our model on the train dataset, and then test it against the test dataset.

    train_data = pd.read_csv("titanic_dataset/train.csv")
    test_data = pd.read_csv("titanic_dataset/test.csv")

    train_data


    # ## EDA

    train_data.describe()

    sns.countplot(x="Survived",data= train_data)
    plt.title("Survival")
    plt.show()


    # ## Boxplots

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


    # ## Histograms

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


    # ## Preprocessing

    print(f"Null values:\n",train_data.isnull().sum())

    print(f"Duplicated data: ",train_data.duplicated().sum())

    print(f"Categorical Columns:\n",[col for col in train_data.columns if train_data[col].dtype=="object"])


    # From the above observation, we can find that 'Age' and 'Cabin' column contain a lot of missing values. For the 'Embarked' column we can either drop them or fill them.

    # Transforming the categorical data in our 'Sex' column to numerical data


    train_data['Sex'] = train_data['Sex'].astype(str).map({'male': 1, 'female': 0})
    train_data['Sex']

    test_data['Sex'] = test_data['Sex'].astype(str).map({'male': 1, 'female': 0})
    test_data['Sex']


    # Handling the missing values in our 'Embarked' column and transforming the categorical data.

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


    # ## Variables

    # Dropping PassengerId, Name, Ticket, Cabin, Embarked column.
    # These columns are either not significant for our model or contain a lot of NaN values.

    train_data.columns


    y = train_data["Survived"]
    x_with = train_data.drop(columns=["Survived", "PassengerId", "Name","Ticket","Cabin","Embarked"],axis=1)

    #dropping Parch and Fare
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


    # Though the difference between the AIC's before and after isn't much, dropping Parch and Fare columns from our feature tends to reduce our AIC for the model.

    # Scaling down our Age and Fare columns


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


    test_data["log_Fare"]=test_data["log_Fare"].fillna(test_data["log_Fare"].median())
    test_data= scaler.transform(test_data)
    preds_on_test = log_model.predict(test_data)
    submission = pd.DataFrame({"PassengerId": test_passengerId,"Survived": preds_on_test})
    submission.to_csv("submission.csv", index=False)

if __name__=="__main__":
    main()