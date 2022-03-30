from typing import List, Optional

from pydantic import BaseModel, Field

from .annotation import Flank
from .base import PyObjectId


class VideoSegmentation(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    examination_id: Optional[PyObjectId]
    annotation: List[Flank] = []
    prediction_raw: List[Flank] = []
    prediction_smooth: List[Flank] = []
    prediction: List[Flank] = []

    def to_dict(self):
        _ = self.dict(exclude_none=True, exclude_defaults=True)
        return _

    class Config:
        allow_population_by_field_name = True
