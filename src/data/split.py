
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(input_path, output_dir):
    # Charger les données
    data = pd.read_csv(input_path)

    # Séparation des variables explicatives et de la cible
    X = data.drop(columns=['silica_concentrate'])
    y = data['silica_concentrate']

    # Split en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Sauvegarder les ensembles
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

if __name__ == "__main__":
    split_data("./data/raw_data/raw.csv", "./data/processed_data")