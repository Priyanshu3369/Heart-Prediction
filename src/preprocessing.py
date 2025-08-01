# src/preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

CATEGORICAL_COLS = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
NUMERIC_COLS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    scaler = StandardScaler()
    df_encoded[NUMERIC_COLS] = scaler.fit_transform(df_encoded[NUMERIC_COLS])

    df_encoded = df_encoded.astype({col: 'int' for col in df_encoded.select_dtypes('bool').columns})
    return df_encoded

def save_processed_data(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    df_clean = preprocess_data(df)
    df_clean.to_csv(output_path, index=False)
    return df_clean

if __name__ == "__main__":
    df_cleaned = save_processed_data("data/raw/heart.csv", "data/processed/heart_cleaned.csv")
    print("✅ Preprocessing complete. Cleaned data shape:", df_cleaned.shape)
