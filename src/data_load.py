import pandas as pd

def load_data(TRAIN_PATH):
    train_data = pd.read_csv(TRAIN_PATH)
    train_data.head()
    train_data.describe()
    return train_data
