import pandas as pd

def load_data(TRAIN_PATH):
    train_data = pd.read_csv(TRAIN_PATH)
    return train_data
