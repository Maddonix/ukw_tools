from datetime import datetime
from typing import List, Union, Optional

from pydantic import BaseModel, Field, validator
from pathlib import Path

from ..classes.annotation import Flank

from datetime import datetime, tzinfo
from .utils import ORIGIN_LOOKUP

INTERVENTION_TYPE_MAPPING = {
    "Coloscopy": "Koloskopie",
    "Unknown": "Unknown",
    "Gastroscopy": "Gastroskopie"
}

class VideoExtern(BaseModel):
    id_extern: int = Field(alias="videoId")
    path: Path = Field(alias= "videoPath")
    images_path: Optional[Path] = Field(alias= "imagesPath")
    annotation_paths: Optional[Path]
    video_remark: Optional[str] #= Field(alias="")
    intervention_histo_text: Optional[str] =  Field(alias = "patho")
    intervention_report_text: Optional[str] = Field(alias = "report")
    origin: Optional[str] = Field(alias = "center")
    intervention_type: str = Field(alias="videoType")
    annotated: bool


    class Config:
        # allow_population_by_field_name = True
        json_encoders = {Path: str}
        schema_extra = {"example": {}}

    @validator('intervention_histo_text')
    def rm_whitespace_histo(cls, v) -> str:
        if v:
            v = v.replace("\n", " ")
            v = v.replace("  ", " ")
            v.strip()
            return v

    @validator('intervention_report_text')
    def rm_whitespace_report(cls, v) -> str:
        if v:
            v = v.replace("\n", " ")
            v = v.replace("  ", " ")
            v.strip()
            return v

    def map_intervention_type(self):
        _type=self.intervention_type
        assert _type in INTERVENTION_TYPE_MAPPING
        _type = INTERVENTION_TYPE_MAPPING[_type]
        return _type

    def examination(self):
        return {
            "id_extern": self.id_extern,
            "path": str(self.path),
            "video_key": self.path.name,
            "is_video": True,
            "origin": self.origin,
            "origin_category": ORIGIN_LOOKUP[self.origin],
            "examination_type": self.map_intervention_type()
        }

    def report(self):
        if hasattr(self, "intervention_histo_text"):
            histo_report = self.intervention_histo_text
        else: histo_report = None

        if hasattr(self, "intervention_report_text"):
            examination_report = self.intervention_report_text
        else: examination_report = None

        report = {
            "examination": examination_report,
            "histo": histo_report,
            "id_extern": self.id_extern,
        }

        return report
    
        # metadata_dict = self.get_video_meta()
        # metadata_dict["path"] = self.path
        # metadata_dict["is_video"] = True
        # metadata_dict["intervention_date"] = path_to_timestamp(self.path, time_format, self.origin)
        # metadata_dict["intervention_type"] = self.map_intervention_type()

        # intervention_dict["metadata"] = metadata_dict

        # return intervention_dict

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
