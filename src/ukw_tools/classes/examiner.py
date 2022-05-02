from pydantic import BaseModel
from .base import PyObjectId, Field
from typing import Optional, List

class Examiner(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    name_unique: str
    center: str
    synonyms: List[str] = []