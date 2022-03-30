import warnings
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from bson import ObjectId
from pymongo import MongoClient
from sklearn.model_selection import train_test_split

from .annotation import ImageAnnotations, MultilabelAnnotation
from .examination import Examination
from .image import Image, ImageCollection
from .model_handler import ModelHandler


def filter_label_array(array, target_labels, labels):
    """Label array of shape (n, m) where n is the number of images and m is the number of labels.\
        Selects columns of target labels and returns a new array of shape (n, len(target_labels))"""

    select = [labels.index(_) for _ in target_labels]
    array = array[:, select]
    return array


class DbHandler:
    def __init__(self, url):
        self.client = MongoClient(url, connectTimeoutMS=200, retryWrites=True)
        self.db = self.client.EndoData3
        self.examination = self.db.Examination
        self.image = self.db.Image
        self.image_collection = self.db.ImageCollection
        self.multilabel_annotation = self.db.MultilabelAnnotation
        self.video_segmentation = self.db.VideoSegmentation
        self.model = self.db.Model

    def clear_all(self):
        print("Deactivated")
        # self.examination.delete_many({})
        # self.image.delete_many({})
        # self.image_collection.delete_many({})

    def insert_frames(self, frames: List[Dict]):
        frame_dict = {}
        for frame in frames:
            r = self.image.insert_one(frame)
            frame_dict[str(frame["n"])] = r.inserted_id

        return frame_dict

    def insert_examination(self, examination, insert_frames=True):
        if examination.is_video:
            assert examination.path.exists()

        fps, frame_count = examination.get_video_info()
        examination.fps = fps
        examination.frame_count = frame_count
        r = self.examination.insert_one(examination.to_dict())
        assert r
        inserted_id = r.inserted_id
        examination.id = inserted_id

        frames = examination.generate_frame_list()
        frames = [Image(**_) for _ in frames]
        frames = [_.to_dict() for _ in frames]
        if insert_frames:
            frame_dict = self.insert_frames(frames)

        self.image_collection.insert_one(
            {"examination_id": examination.id, "type": "frame", "images": frame_dict}
        )

    def insert_model(self, model: ModelHandler):
        model_dict = model.to_dict(exclude_none=True)
        if "_id" in model_dict:
            r = self.model.update_one({"_id": model_dict["_id"]}, {"$set": model_dict})
        else:
            r = self.model.insert_one(model_dict)

        return r

    def update_examination_crop(self, _id: ObjectId, crop: Tuple[int]):
        self.examination.update_one({"_id": _id}, {"$set": {"crop": crop}})

    def update_frames_extracted(
        self, examination_id: ObjectId, frame_dict: Dict[int, Path]
    ):
        image_dict = self.image_collection.find_one(
            {"examination_id": examination_id, "type": "frame"}
        )
        for n, path in frame_dict.items():
            self.image.update_one(
                {"_id": image_dict["images"][str(n)]},
                {"$set": {"path": path.as_posix(), "is_extracted": True}},
            )

    def get_multilabel_train_data(self, test_size = 0.1):
        img_ids = [
            _["image_id"] for _ in self.multilabel_annotation.find({})
        ]
        train, val = train_test_split(img_ids, test_size=test_size)
        train = {i: _id for i, _id in enumerate(train)}
        val = {i: _id for i, _id in enumerate(val)}
        train_collection = ImageCollection(type="multilabel_train", images=train)
        train_collection_id = self.image_collection.insert_one(
            train_collection.to_dict()
        ).inserted_id

        val_collection = ImageCollection(type="multilabel_val", images=val)
        val_collection_id = self.image_collection.insert_one(
            val_collection.to_dict()
        ).inserted_id

        return train_collection_id, val_collection_id

    def get_examination(self, _id: ObjectId, as_object=True):
        examination = self.examination.find_one({"_id": _id})
        if as_object:
            examination = Examination(**examination)

        return examination

    def get_examinations(self, ids: List[ObjectId], as_object=True):
        examinations = self.examination.find({"_id": {"$in": ids}})
        if as_object:
            examinations = [Examination(**_) for _ in examinations]
        else:
            examinations = [_ for _ in examinations]

        return examinations

    def get_image(self, _id: ObjectId, as_object=True):
        image = self.image.find_one({"_id": _id})
        if as_object:
            image = Image(**image)

        return image

    def get_images(self, ids: List[ObjectId], as_object=True):
        images = self.image.find({"_id": {"$in": ids}})
        if as_object:
            images = [Image(**_) for _ in images]
        else:
            images = [_ for _ in images]

        return images

    def get_model_settings(self, model_id:ObjectId):
        settings = self.model.find_one({"_id": model_id})
        trainer_settings = settings["trainer"]
        model_settings = settings["model"]

        return model_settings, trainer_settings

    def prepare_ds_from_image_collection(self, image_collection_id, predict=False, target_labels=None):
        image_collection = self.image_collection.find_one({"_id": image_collection_id})
        paths = []
        _ids = []
        labels = []
        crop = []
        choices = []

        if image_collection:
            images = self.image.find(
                {
                    "_id": {"$in": list(image_collection["images"].values())},
                    "is_extracted": True,
                }
            )
            # ids = [_["_id"] for _ in images]
            examination_ids = self.image.distinct(
                "examination_id",
                {"_id": {"$in": list(image_collection["images"].values())}},
            )
            lookup_crop = {}
            for examination_id in examination_ids:
                _ = self.examination.find_one({"_id": examination_id})
                if "crop" in _:
                    lookup_crop[examination_id] = _["crop"]
                else:
                    warnings.warn(f"Examination {examination_id} has no crop")

            for image in images:
                if image["examination_id"] in lookup_crop:
                    crop.append(lookup_crop[image["examination_id"]])
                    paths.append(image["path"])
                    _ids.append(image["_id"])
                else:
                    continue

        if predict:
            labels = [str(_) for _ in _ids]
        else:
            assert target_labels

            for _id in _ids:
                annotation = self.get_multilabel_image_annotation(_id)
                annotation = annotation.latest_annotation()
                if not choices:
                    choices = annotation.choices
                else:
                    assert choices == annotation.choices

                labels.append(annotation.value)

            label_records = []
            for label in labels:
                record = []
                for i, _ in enumerate(choices):
                    if i in label:
                        record.append(1.0)
                    else: record.append(0.0)
                label_records.append(record)

            labels = np.array(label_records, dtype = float)
            labels = filter_label_array(labels, target_labels, choices)

        return paths, labels, crop, choices

    def get_examination_image_collection(self, examination_id: ObjectId):
        image_collection = self.image_collection.find_one(
            {"examination_id": examination_id}
        )
        return ImageCollection(**image_collection)

    def get_examination_frame_dict(self, examination_id: ObjectId):
        image_collection = self.image_collection.find_one(
            {"examination_id": examination_id, "type": "frame"}
        )
        if image_collection:
            return image_collection["images"]

        return {}

    def get_multilabel_image_annotation(self, image_id: ObjectId):
        image = self.multilabel_annotation.find_one({"image_id": image_id})
        if image:
            return ImageAnnotations(**image)

        return {}

    def get_multilabel_image_annotations(self, image_ids: List[ObjectId]):
        images = self.multilabel_annotation.find({"image_id": {"$in": image_ids}})
        images = [ImageAnnotations(**image) for image in images]

        return images

    def set_image_annotation(
        self, image_id: ObjectId, annotation: MultilabelAnnotation
    ):
        r = self.multilabel_annotation.update_one(
            {"_id": image_id},
            {"$set": {str(annotation.annotator_id): annotation.to_dict()}},
        )
        assert r.matched_count == 1
        return r

    # def get_report(self, _id: ObjectId, as_object = True):
    #     report = self.ASDASDASDASDASD.find_one({"examination_id": _id, "type": "report"})
    #     if as_object:
    #         report = ImageCollection(**report)

    #     return report
