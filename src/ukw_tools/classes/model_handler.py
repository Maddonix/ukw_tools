from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..model.multilabel_classification_net import MultilabelClassificationNet
from .base import PyObjectId
from .image import ImageCollection

MODEL_LOOKUP = {
    "colo_segmentation": MultilabelClassificationNet
}

class ModelHandler(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    date: datetime = datetime.now()
    path: Optional[Path]
    model_name: str # e.g. colo_segmentation
    model_type: str # e.g. EfficientNet-B4
    labels: Optional[List[str]]
    train_images: Optional[ImageCollection]
    val_images: Optional[ImageCollection]
    model_settings: Optional[Dict[str, str]]
    metrics: Any

    """
    Example model_settings:
    {
        labels = ["background", "foreground"],
        lr=6e-3,
        wigth_decay=1e-5,
        pos_weight=2,
        model_type="RegNetX800MF"
    }
    """

    class Config:
        allow_population_by_field_name = True

    def load_model_from_checkpoint(self, gpu = True, eval = True):
        assert hasattr(self, "path"), "ModelHandler has no path attribute"

        model_class = MODEL_LOOKUP[self.model_name]
        model = model_class.load_from_checkpoint(self.path)
        if gpu:
            model.cuda()
        if eval:
            model.eval()

        return model

    def prepare_model(self, gpu = True, eval = False):
        assert hasattr(self, "model_settings"), "ModelHandler has no model_settings attribute"

        model_class = MODEL_LOOKUP[self.model_name]
        model = model_class(**self.model_settings)
        if gpu:
            model.cuda()
        if eval:
            model.eval()

        return model
