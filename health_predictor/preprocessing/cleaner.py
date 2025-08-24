# cleaner.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    df_encoded = df.copy()
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == "object":
            le = LabelEncoder()
            df_encoded[column] = le.fit_transform(df[column])
            label_encoders[column] = le
    return df_encoded, label_encoders
