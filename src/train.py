import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
import os

def load_clean_data(path: str):
    df = pd.read_csv(path)
    X = df.drop('target', axis=1)
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }

def train_and_evaluate_baselines():
    X_train, X_test, y_train, y_test = load_clean_data("data/processed/heart_cleaned.csv")

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier()
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_test, y_test)

    return results

if __name__ == "__main__":
    metrics = train_and_evaluate_baselines()

    for model_name, scores in metrics.items():
        print(f"🔍 {model_name}")
        for metric, value in scores.items():
            print(f"   {metric}: {value:.4f}")

    os.makedirs("outputs/reports", exist_ok=True)
    with open("outputs/reports/baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
