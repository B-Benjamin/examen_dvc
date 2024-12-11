import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

def train_model(input_dir, model_dir):
    # Charger les données et les meilleurs paramètres
    X_train = pd.read_csv(f"{input_dir}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{input_dir}/y_train.csv").squeeze()
    best_params = joblib.load(f"{model_dir}/best_params.pkl")

    # Entraîner le modèle
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)

    # Sauvegarder le modèle
    joblib.dump(model, f"{model_dir}/trained_model.pkl")

if __name__ == "__main__":
    train_model("./data/processed_data", "./models")
