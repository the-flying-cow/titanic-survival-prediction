import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def pre_processing(train_data):
    print(f"Null values:\n",train_data.isnull().sum())
    print(f"Duplicated data: ",train_data.duplicated().sum())
    print(f"Categorical Columns:\n",[col for col in train_data.columns if train_data[col].dtype=="object"])

    train_data['Sex'] = train_data['Sex'].astype(str).map({'male': 1, 'female': 0})
    train_data['Sex']

    print(train_data['Embarked'].mode())
    
    train_data['Embarked'].dtype
    train_data['Embarked']= train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
    embarked_dummies = pd.get_dummies(train_data['Embarked'],prefix='Embarked',drop_first=True).astype(int)
    train_data = pd.concat([train_data,embarked_dummies], axis=1)

    train_data["Age"] = train_data["Age"].fillna(train_data["Age"].median())
    train_data["log_Fare"] = np.log1p(train_data["Fare"])
    
    num_data = train_data.drop(columns=["PassengerId","Survived","Name","Ticket","Cabin","Embarked"],axis=1)
    corr_matrix = num_data.corr()
    plt.figure(figsize=(10,10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    return train_data