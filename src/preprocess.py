import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_train(df):
    df = df.select_dtypes(include=[np.number])
    df = df.ffill().bfill()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    return X_scaled, scaler

def preprocess_test(df, scaler):
    df = df.select_dtypes(include=[np.number])
    df = df.ffill().bfill()

    X_scaled = scaler.transform(df)
    return X_scaled
