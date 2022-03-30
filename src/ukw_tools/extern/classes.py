from datetime import datetime
from typing import List, Union

from pydantic import BaseModel, Field, validator

from ..classes.annotation import Flank


class ExternFlank(BaseModel):
    name: str = Field(alias="annotationName")
    value: Union[bool, str] = Field(alias="annotationValue")
    start: int = Field(alias="startFrame")
    stop: int = Field(alias="endFrame")

    @validator("name")
    def name_to_lower(cls, v):
        return v.lower()

    def to_intern(self, source, annotator_id, date, instance_id=None):
        return Flank(
            source=source,
            annotator_id=annotator_id,
            date=date,
            name=self.name,
            value=self.value,
            start=self.start,
            stop=self.stop,
            instance_id=instance_id,
        )


class ExternAnnotatedVideo(BaseModel):
    id_extern: int = Field(alias="videoId")
    video_key: str = Field(alias="videoName")
    date: datetime = Field(alias="lastAnnotationSession")

    @validator("date")
    def rm_tz(cls, v):
        v = v.replace(tzinfo=None)
        v = v.replace(microsecond=0)
        return v


class ExternVideoFlankAnnotation(BaseModel):
    extern_annotator_id: int = Field(alias="userId")
    id_extern: int = Field(alias="videoId")
    video_key: str = Field(alias="videoName")
    date: datetime = Field(alias="sessionDate")
    label_group_id: int = Field(alias="labelGroupId")
    label_group_name: str = Field(alias="labelGroupName")
    flanks: List[ExternFlank] = Field(alias="sequences")

    @validator("date")
    def rm_tz(cls, v):
        v = v.replace(microsecond=0)
        return v
