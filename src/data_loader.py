import pandas as pd

def load_local_data(path: str) -> pd.DataFrame:
    path = "../data/raw/heart.csv"
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    path = "data/raw/heart.csv"
    df = load_local_data(path)
    
    print("✅ Dataset downloaded and saved.")
    print("🔢 Shape:", df.shape)
    print("🧾 Columns:", df.columns.tolist())
    print(df.head())
