import os
from enum import StrEnum

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

app = FastAPI()


# Use enums for validating categorical input data
class PropertyType(StrEnum):
    DEPARTAMENTO = "departamento"
    CASA = "casa"


class Sector(StrEnum):
    LAS_CONDES = "las condes"
    VITACURA = "vitacura"
    LA_REINA = "la reina"
    LO_BARNECHEA = "lo barnechea"
    PROVIDENCIA = "providencia"
    NUNOA = "nunoa"


# Define the input data model
class InputData(BaseModel):
    type: PropertyType
    sector: Sector
    net_usable_area: float
    net_area: float
    n_rooms: int
    n_bathroom: int
    latitude: float
    longitude: float

    @field_validator("sector", "type", mode="before")
    def validate_str(cls, value):
        return str(value).lower().strip()


# Load the pre-trained sklearn pipeline
model_path = os.environ.get("PROPERTY_MODEL_PATH", "model.pkl")
try:
    pipeline = joblib.load(model_path)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Model not found at {model_path}. Please set the PROPERTY_MODEL_PATH"
        "environment variable adequately."
    )


@app.post("/predict")
def predict(data: InputData):
    try:
        # maybe there's a better way to do this instead of using a DataFrame,
        # but it seems fast enough
        input_df = pd.DataFrame([data.model_dump()])

        return pipeline.predict(input_df)[0]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
