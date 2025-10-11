import pandas as pd
import numpy as np


def pre_processing(train_data):
    train_data['Sex'] = train_data['Sex'].astype(str).map({'male': 1, 'female': 0})
    train_data['Sex']

    train_data['Embarked'].dtype
    train_data['Embarked']= train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
    embarked_dummies = pd.get_dummies(train_data['Embarked'],prefix='Embarked',drop_first=True).astype(int)
    train_data = pd.concat([train_data,embarked_dummies], axis=1)

    train_data["Age"] = train_data["Age"].fillna(train_data["Age"].median())
    train_data["log_Fare"] = np.log1p(train_data["Fare"])

    return train_data