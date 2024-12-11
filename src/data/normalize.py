import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_data(input_dir, output_dir):
    # Charger les ensembles
    X_train = pd.read_csv(f"{input_dir}/X_train.csv")
    X_test = pd.read_csv(f"{input_dir}/X_test.csv")

    # Supprimer la colonne de date dans les deux ensembles
    X_train = X_train.drop(columns=['date'])
    X_test = X_test.drop(columns=['date'])
    
    # Appliquer la normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Sauvegarder les données normalisées
    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(f"{output_dir}/X_train_scaled.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(f"{output_dir}/X_test_scaled.csv", index=False)

if __name__ == "__main__":
    normalize_data("./data/processed_data", "./data/processed_data")