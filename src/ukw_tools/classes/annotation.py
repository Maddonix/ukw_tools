from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, NonNegativeInt, validator

from .base import PyObjectId


class Annotation(BaseModel):
    source: str  # constr(regex=r"")
    annotator_id: NonNegativeInt
    date: datetime
    name: str
    instance_id: Optional[int]

    def __hash__(self):
        return hash(repr(self))

    @validator("date")
    def rm_ms(cls, v):
        v = v.replace(microsecond=0)
        return v


class MultilabelAnnotation(Annotation):
    value: List[NonNegativeInt]
    choices: List[str]
    instance_id: Optional[PyObjectId]


class ImageAnnotations(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    image_id: Optional[PyObjectId]
    examination_id: Optional[PyObjectId]
    annotations: Dict[int, MultilabelAnnotation] = {}

    class Config:
        allow_population_by_field_name = True

    def latest_annotation(self):
        date = None
        for key, value in self.annotations.items():
            if not date:
                date = value.date
                selected_key = key
            if value.date > date:
                date = value.date
                selected_key = key

        return self.annotations[selected_key]

    def to_dict(self):
        _ = self.dict(exclude_none=True)
        _["annotations"] = {str(k): v for k, v in _["annotations"].items()}
        return _


class Flank(Annotation):
    start: int
    stop: int
    value: Union[bool, str]
