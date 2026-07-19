from pydantic import BaseModel, Field
from typing import Literal

Language = Literal["mr", "hi", "en"]

class FarmerRegistration(BaseModel):
    farmerId: str = Field(min_length=3, max_length=100)
    name: str = Field(min_length=1, max_length=120)
    phone: str = Field(min_length=8, max_length=20)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    crops: list[str] = Field(min_length=1)
    preferredLanguage: Language = "mr"

class PredictionResponse(BaseModel):
    scanId: str
    className: str
    confidence: float
    crop: str
    disease: str
    localized: dict
    treatment: dict
    heatmapBase64: str | None = None
    voiceUrl: str | None = None

class ScanHistoryItem(BaseModel):
    timestamp: str
    className: str
    confidence: float
    crop: str
    disease: str
    thumbnailBase64: str | None = None

