import logging
import os

from fastapi.security import APIKeyHeader
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi import Depends
from property_pipeline_bain.api.models import InputData

logger = logging.getLogger(__name__)

app = FastAPI()

API_KEY_NAME = "Bain-API-Key"
API_KEY = os.environ.get("BAIN_API_KEY")

# Load the pre-trained sklearn pipeline
model_path = os.environ.get("PROPERTY_MODEL_PATH", "model.pkl")
try:
    pipeline = joblib.load(model_path)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Model not found at {model_path}. Please set the PROPERTY_MODEL_PATH"
        "environment variable adequately."
    )


api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)


def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key is None or api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


@app.post("/predict")
def predict(data: InputData, api_key: str = Depends(verify_api_key)):
    try:
        # maybe there's a better way to do this instead of using a DataFrame,
        # but it seems fast enough
        print(data)
        model_dump = data.model_dump()
        logger.info(f"Received data: {model_dump}")
        input_df = pd.DataFrame([model_dump])

        prediction = pipeline.predict(input_df)[0]
        logger.info(f"Prediction: {prediction}")
        return prediction

    except Exception as e:
        logger.exception(f"An error occurred while predicting: {e}.")
        raise HTTPException(status_code=500, detail=str(e))
