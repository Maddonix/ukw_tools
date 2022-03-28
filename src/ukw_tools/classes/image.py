from typing import (
    Dict,
    Optional
)
from pathlib import Path
from pydantic import (
    BaseModel,
    Field,
)

from .base import PyObjectId

class Image(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    intervention_id: PyObjectId
    origin: str
    origin_category: str
    image_type: str # freeze, frame
    n: int
    is_extracted: bool
    annotation: Optional[PyObjectId]
    prediction: Optional[PyObjectId]
    path: Optional[Path]

    class Config:
        allow_population_by_field_name = True
        # arbitrary_types_allowed = True
        json_encoders = {Path: str}

    def to_dict(self):
        _ = self.dict(exclude_none=True)
        if self.path:
            _["path"] = str(self.path)
        return _

class ImageCollection(BaseModel):
    examination_id: PyObjectId
    images: Dict[int, PyObjectId]