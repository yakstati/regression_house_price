import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data/raw/test.csv'
PIPELINE_PATH = BASE_DIR / 'models/final_pipeline.joblib'

def main():
    X = pd.read_csv(DATA_PATH)

    pipeline = joblib.load(PIPELINE_PATH)

    predictions = pipeline.predict(X)

    result = pd.DataFrame({
        'Id': X['Id'],
        'pred': predictions
    })

    result.to_csv(BASE_DIR / "result.csv", index=False)

if __name__ == "__main__":
    main()