import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

def evaluate_model(input_dir, model_dir, output_dir):
    # Charger les données et le modèle
    X_test = pd.read_csv(f"{input_dir}/X_test_scaled.csv")
    y_test = pd.read_csv(f"{input_dir}/y_test.csv").squeeze()
    model = joblib.load(f"{model_dir}/trained_model.pkl")

    # Faire des prédictions
    y_pred = model.predict(X_test)

    # Calculer les métriques
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Sauvegarder les scores
    scores = {"mse": mse, "r2": r2}
    with open(f"{output_dir}/scores.json", "w") as f:
        json.dump(scores, f)

    # Sauvegarder les prédictions
    predictions = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    predictions.to_csv(f"{output_dir}/predictions.csv", index=False)

if __name__ == "__main__":
    evaluate_model("./data/processed_data", "./models", "./metrics")