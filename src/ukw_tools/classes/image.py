from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field

from .base import PyObjectId


class Image(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    examination_id: PyObjectId
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

    def exists(self):
        if self.is_extracted:
            return self.path.exists()
        else:
            return False

class ImageCollection(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    examination_id: Optional[PyObjectId]
    model_id: Optional[PyObjectId]
    type: Optional[str]
    images: Dict[int, PyObjectId]
    date: Optional[datetime] = datetime.now()

    def to_dict(self):
        _ = self.dict(exclude_none=True)
        _["images"] = {str(k):v for k,v in _["images"].items()}
        return _
