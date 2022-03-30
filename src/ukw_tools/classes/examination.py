from typing import (
    List,
    Optional,
    Tuple
)
from pathlib import Path
from pydantic import (
    BaseModel,
    Field,
)

from .base import PyObjectId
from ..media.video import get_video_info

class Examination(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    origin: str
    origin_category: str
    examination_type: str
    is_video: bool
    video_key: Optional[str]
    id_extern: Optional[int]
    examiners: Optional[List[str]]
    age: Optional[int]
    gender: Optional[int]
    path: Optional[Path]
    fps: Optional[int]
    frame_count: Optional[int]
    frames: Optional[PyObjectId]
    freezes: Optional[PyObjectId]
    report: Optional[PyObjectId]
    segmentation: Optional[PyObjectId]
    annotation: Optional[PyObjectId]
    prediction: Optional[PyObjectId]
    crop: Optional[Tuple[int, int, int, int]] # ymin, ymax, xmin, xmax

    class Config:
        allow_population_by_field_name = True
        # arbitrary_types_allowed = True
        json_encoders = {Path: str}

    def to_dict(self):
        _dict = self.dict(exclude_none=True)
        _dict["path"] = str(self.path)

        return _dict

    def get_video_info(self):
        """
        Get video info from a video file.

        Returns fps, frame_count
        """
        return get_video_info(self.path)

    def get_frame_template(self):
        template = {
            "examination_id": self.id,
            "origin": self.origin,
            "origin_category": self.origin_category,
            "image_type": "frame",
            "is_extracted": False
        }

        return template

    def generate_frame_list(self):
        """
        Generate a list of frames from a video.
        """
        template = self.get_frame_template()
        frames = []
        self.fps, self.frame_count = self.get_video_info()
        for i in range(self.frame_count):
            _ = template.copy()
            _["n"] = i
            frames.append(_)

        return frames