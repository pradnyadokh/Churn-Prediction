import json
import pickle
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_prediction_model.pkl"
COLUMNS_PATH = PROJECT_ROOT / "models" / "columns.json"

if not MODEL_PATH.exists() and (PROJECT_ROOT / "end_to_end_deployment" / "models" / "churn_prediction_model.pkl").exists():
    MODEL_PATH = PROJECT_ROOT / "end_to_end_deployment" / "models" / "churn_prediction_model.pkl"

if not COLUMNS_PATH.exists() and (PROJECT_ROOT / "end_to_end_deployment" / "models" / "columns.json").exists():
    COLUMNS_PATH = PROJECT_ROOT / "end_to_end_deployment" / "models" / "columns.json"


CHURN_THRESHOLD = 0.4
CATEGORICAL_FIELDS = {"gender", "maritalstatus"}


class ChurnPredictor:
    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        if not COLUMNS_PATH.exists():
            raise FileNotFoundError(f"Columns file not found at: {COLUMNS_PATH}")

        with MODEL_PATH.open("rb") as model_file:
            self.model = pickle.load(model_file)

        with COLUMNS_PATH.open("r", encoding="utf-8") as columns_file:
            self.data_columns = json.load(columns_file)["data_columns"]

    def _prepare_input_array(self, customer_data: dict) -> np.ndarray:
        normalized = {}
        for key, value in customer_data.items():
            if isinstance(value, str):
                normalized[key] = value.lower().replace(" ", "_")
            else:
                normalized[key] = value

        input_array = np.zeros(len(self.data_columns))

        for index, column in enumerate(self.data_columns):
            if column in normalized and column not in CATEGORICAL_FIELDS:
                input_array[index] = normalized[column]

        for key, value in normalized.items():
            if key in CATEGORICAL_FIELDS:
                one_hot_column = f"{key}_{value}"
                if one_hot_column in self.data_columns:
                    input_array[self.data_columns.index(one_hot_column)] = 1

        return input_array

    def predict(self, customer_data: dict) -> tuple[str, float]:
        input_array = self._prepare_input_array(customer_data)
        churn_probability = float(self.model.predict_proba([input_array])[0][1])
        churn_probability = round(churn_probability, 4)
        prediction = "Churn" if churn_probability > CHURN_THRESHOLD else "Not Churn"
        return prediction, churn_probability
