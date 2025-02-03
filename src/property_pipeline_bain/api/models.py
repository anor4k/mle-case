# Use enums for validating categorical input data
from enum import StrEnum

from pydantic import BaseModel, field_validator


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
