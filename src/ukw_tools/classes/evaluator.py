from pydantic import BaseModel, Field, validator
from .base import PyObjectId
from typing import Optional

class Evaluator(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    examination_id: Optional[PyObjectId]
    annotation_id: Optional[PyObjectId]
    prediction_id: Optional[PyObjectId]
    
