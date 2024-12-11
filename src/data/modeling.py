import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib

def grid_search(input_dir, output_dir):
    # Charger les données
    X_train = pd.read_csv(f"{input_dir}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{input_dir}/y_train.csv").squeeze()

    # Définir le modèle et les paramètres à tester
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    # Effectuer la recherche par grille
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Sauvegarder les meilleurs paramètres
    best_params = grid_search.best_params_
    joblib.dump(best_params, f"{output_dir}/best_params.pkl")

if __name__ == "__main__":
    grid_search("./data/processed_data", "./models")