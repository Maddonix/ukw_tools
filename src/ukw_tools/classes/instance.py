from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any, Tuple
from .base import PyObjectId

class Instance(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    segmentation_prediction_id: Optional[PyObjectId]
    examination_id: PyObjectId
    label: str
    flank_framenumber_tuples: List[Tuple[int, int]]=[]
    flank_frame_id_tuples: Optional[List[Tuple[PyObjectId, PyObjectId]]]
    keyframe_ids: List[PyObjectId] = []
    keyframe_framenumbers: List[int] = []
