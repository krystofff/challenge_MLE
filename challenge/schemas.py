from typing import List
from pydantic import BaseModel, Field


class FlightIn(BaseModel):
    OPERA: str = Field(..., description="Operator/airline. Example: 'Copa Air'.")
    TIPOVUELO: str = Field(..., description="Flight type: 'I' or 'N'.")
    MES: int = Field(..., description="Month as integer (1..12).")


class PredictRequest(BaseModel):
    flights: List[FlightIn]


class PredictResponse(BaseModel):
    predict: List[int]
