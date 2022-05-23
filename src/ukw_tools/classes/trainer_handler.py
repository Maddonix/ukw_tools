# train_ds, val_ds, train_collection_id, val_collection_id = db.get_multilabel_train_data()
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning
from pydantic import BaseModel
from torch.utils.data import DataLoader
from ..model import wandb_callbacks
from ..model.utils import instantiate

from ..dataset.image_classification import ImageClassificationDs
from .base import PyObjectId
from .db import DbHandler


class TrainerHandler(BaseModel):
    model_name: str = "colo_segmentation"
    model_type: str = "RegNetX800MF"

    target_labels: List[str]

    train_collection_id: PyObjectId
    val_collection_id: PyObjectId

    train_bs: int
    val_bs: int

    train_workers: int
    val_workers: int

    scaling: int

    checkpoint_dir: Path

    gpus: int = 1

    log_dir: Path
    log_freq: int = 100

    save_top_k: int = 1
    monitor: str = "val/loss"
    mode: str = "min"
    early_stopping_patience: int = 25
    max_epochs: int = 300

    trainer_precision:int = 16

    train_ds: Any = None
    val_ds: Any = None
    train_dl: Any = None
    val_dl: Any = None
    model_path: Optional[Path] = None

    callbacks: Dict[str, Any] = {
        "model_checkpoint": {
            "_target_": pytorch_lightning.callbacks.ModelCheckpoint,
            "monitor": None,
            "save_top_k": None,
            "save_last": False,
            "mode": None,
            "dirpath": None,
            "filename": None,
        },
        "early_stopping": {
            "_target_": pytorch_lightning.callbacks.EarlyStopping,
            "monitor": None,
            "patience": None,
            "mode": None,
        },
        "watch_model_with_wandb": {
            "_target_": wandb_callbacks.WatchModelWithWandb,
            "log": "all",
            "log_freq": None,
        },
        "image_prediction_logger": {
            "_target_": wandb_callbacks.ImagePredictionLogger,
            "num_samples": 10
        }
    }

    def get_dataloaders(self, db:DbHandler):
        train_dl = self.prepare_train_data(db)
        val_dl = self.prepare_val_data(db)

        return train_dl, val_dl

    def get_settings(self):
        settings = self.dict(exclude_none = True, exclude= {
            "train_ds",
            "val_ds",
            "train_dl",
            "val_dl",
            "model_filename",
            "callbacks"
        })
        if "model_path" in settings:
            settings["model_path"] = self.model_path.as_posix()
        if "log_dir" in settings:
            settings["log_dir"] = self.log_dir.as_posix()
        if "checkpoint_dir" in settings:
            settings["checkpoint_dir"] = self.checkpoint_dir.as_posix()
        return settings

    def prepare_trainer(self):
        logger, callbacks = self.instantiate_callbacks()
        trainer = pytorch_lightning.Trainer(
            max_epochs=self.max_epochs,
            gpus=self.gpus,
            logger=logger,
            callbacks=callbacks,
            precision=self.trainer_precision,
        )
        settings = self.get_settings()
        return trainer, settings

    def get_model_filename(self):
        i = 0
        filename = f"{self.model_name}_{self.model_type}_{i}.ckpt"
        path = self.checkpoint_dir.joinpath(filename)
        exists = path.exists()
        while exists:
            i += 1
            filename = f"{self.model_name}_{self.model_type}_{i}.ckpt"
            path = self.checkpoint_dir.joinpath(filename)
            exists = path.exists()
                
        self.model_path = path
        return filename

    def dump_config(self):
        pass

    def prepare_callbacks(self):
        self.get_model_filename()
        
        callbacks = self.callbacks.copy()
        callbacks["model_checkpoint"]["dirpath"] = self.checkpoint_dir
        callbacks["model_checkpoint"]["filename"] = self.model_path.with_suffix("").name
        callbacks["model_checkpoint"]["monitor"] = self.monitor
        callbacks["model_checkpoint"]["save_top_k"] = self.save_top_k
        callbacks["model_checkpoint"]["mode"] = self.mode

        callbacks["early_stopping"]["monitor"] = self.monitor
        callbacks["early_stopping"]["patience"] = self.early_stopping_patience
        callbacks["early_stopping"]["mode"] = self.mode

        callbacks["watch_model_with_wandb"]["log_freq"] = self.log_freq

        self.callbacks = callbacks
        return callbacks

    def instantiate_callbacks(self, include_wandb=True):
        if include_wandb:
            logger = {
                "_target_": pytorch_lightning.loggers.wandb.WandbLogger,
                "project": self.model_name,
                "job_type": "train",
                "group": "",
                "save_dir": self.log_dir,
            }
            logger = instantiate(logger)
        else: logger = None

        callbacks = self.prepare_callbacks()
        skip_callbacks = []
        if not include_wandb: 
            skip_callbacks.append("watch_model_with_wandb")
            skip_callbacks.append("image_prediction_logger")

        callbacks = [instantiate(v) for k, v in callbacks.items()]
        return logger, callbacks


    def prepare_train_data(self, db):
        paths, labels, crop, choices = db.prepare_ds_from_image_collection(
            self.train_collection_id,
            predict = False,
            target_labels = self.target_labels,
            )
        self.train_ds = ImageClassificationDs(
            paths = paths, 
            labels = labels,
            crop = crop,
            scaling = self.scaling,
            training = True,
        )
        self.train_dl = DataLoader(self.train_ds, batch_size = self.train_bs, num_workers = self.train_workers, shuffle = True)
        return self.train_dl

    def prepare_val_data(self, db):
        paths, labels, crop, choices = db.prepare_ds_from_image_collection(
            self.val_collection_id,
            predict = False,
            target_labels = self.target_labels,
            )
        self.val_ds = ImageClassificationDs(
            paths = paths, 
            labels = labels,
            crop = crop,
            scaling = self.scaling,
            training = True,
        )
        self.val_dl = DataLoader(self.val_ds, batch_size = self.val_bs, num_workers = self.val_workers, shuffle = False)
        return self.val_dl
