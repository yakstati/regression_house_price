import pandas as pd
import joblib
from catboost import CatBoostRegressor
from pathlib import Path

from src.transformers import DataCleaner, LifesquareKNNImputer, HealthcareKNNImputer, FeaturesCreator
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data/raw/test.csv'
MODEL_PATH = BASE_DIR / 'models/catboost_model.cbm'
PREPROCESSOR_PATH = BASE_DIR / 'models/preprocessor.joblib'

def main():
    X = pd.read_csv(DATA_PATH)

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    cb_model = CatBoostRegressor()
    cb_model.load_model(str(MODEL_PATH))

    X_prep = preprocessor.transform(X)
    predictions = cb_model.predict(X_prep)

    result = pd.DataFrame({
        'Id': X['Id'],
        'pred': predictions
    })

    result.to_csv(BASE_DIR / "result.csv", index=False)

if __name__ == "__main__":
    main()