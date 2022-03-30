from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
from .base import PyObjectId

class Prediction(BaseModel):
    model_id: PyObjectId
    image_id: PyObjectId
    prediction: List[float]
    choices: List[str]
    labels: List[List[str]]

    