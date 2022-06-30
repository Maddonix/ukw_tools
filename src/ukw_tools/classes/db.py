import warnings
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from bson import ObjectId
from pymongo import MongoClient
from sklearn.model_selection import train_test_split

from ..media.video import get_frame_dir, get_frame_name
from .annotation import ImageAnnotations, MultilabelAnnotation
from .examination import Examination
from .image import Image, ImageCollection
from .model_handler import ModelHandler
from .prediction import Prediction, VideoSegmentPrediction
from ..extern.requests import get_extern_examinations
from .report import Report
from .video_segmentation import VideoSegmentation
import warnings
from .evaluator import Evaluator
from .instance import Instance


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
        self.examination_dashboard_data = self.db.ExaminationDashboardData
        self.image = self.db.Image
        self.image_collection = self.db.ImageCollection
        self.multilabel_annotation = self.db.MultilabelAnnotation
        self.multilabel_prediction = self.db.MultilabelPrediction
        self.video_segmentation = self.db.VideoSegmentation
        self.video_segmentation_prediction = self.db.VideoSegmentationPrediction
        self.report = self.db.Report
        self.model = self.db.Model
        self.evaluator = self.db.Evaluator
        self.examiner = self.db.Examiner
        self.freeze_detection = self.db.FreezeDetection
        self.instance = self.db.Instance
        self.sequence = self.db.Sequence

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
        self, examination_id: ObjectId, frame_list: List[int]
    ):
        examination = self.get_examination(examination_id)
        frame_dir = get_frame_dir(examination.video_key)
        image_dict = self.image_collection.find_one(
            {"examination_id": examination_id, "type": "frame"}
        )
        for n in frame_list:
            path = frame_dir.joinpath(get_frame_name(n))
            self.image.update_one(
                {"_id": image_dict["images"][str(n)]},
                {"$set": {"path": path.as_posix(), "is_extracted": True}},
            )

    def get_multilabel_train_data(self, test_size = 0.1, exclude_examination_ids = None):
        if not exclude_examination_ids: exclude_examination_ids = []
        img_ids = [
            _["image_id"] for _ in self.multilabel_annotation.find(
                {"examination_id": {"$nin": exclude_examination_ids},
                "annotations": {
                    "$nin": [
                        None,
                        [],
                        {}
                    ]
                },
                }
            )
        ]
        train, val = train_test_split(img_ids, test_size=test_size)
        print("GENERATED DATASETS")
        print(f"Train: {len(train)}")
        print(f"Val: {len(val)}")
        
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

    def get_image_number(self, _id: ObjectId):
        image = self.image.find_one({"_id": _id})
        return image["n"]

    def get_model_settings(self, model_id:ObjectId):
        settings = self.model.find_one({"_id": model_id})
        trainer_settings = settings["trainer"]
        model_settings = settings["model"]

        return model_settings, trainer_settings

    def describe_image_collection(self, collection_id: ObjectId):
        collection = self.image_collection.find_one({"_id": collection_id})
        images = self.image.find({"_id": {"$in": list(collection["images"].values())}})
        images = [_ for _ in images]
        df = pd.DataFrame.from_records(images)

        r = {
            "n": len(df),
            "origin": df.origin.value_counts().to_dict(),
            "origin_category": df.origin_category.value_counts().to_dict(),
            "n_examinations": len(list(df.examination_id.unique())),
        }

        return r

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

    def get_examination_instances(self, examination_id: ObjectId):
        instances = self.instance.find({"examination_id": examination_id})
        if instances:
            instances = [Instance(**_) for _ in instances]
        else:
            instances = []
        return instances

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

    def get_examination_segmentation_annotation(self, examination_id:ObjectId, as_object = True):
        _ = self.video_segmentation.find_one({"examination_id": examination_id})
        if _:
            if as_object:
                return VideoSegmentation(**_)
            else:
                return _

    def get_examination_segmentation_prediction(self, examination_id:ObjectId, as_object = True):
        try:
            _ = self.video_segmentation_prediction.find_one({"examination_id": examination_id})
            if _:
                for k, v in _["predicted_times"].items():
                    if isinstance(v, list):
                        _["predicted_times"][k] = None

                if as_object:
                    return VideoSegmentPrediction(**_)
                else:
                    return _

            else:
                _ = VideoSegmentPrediction(examination_id=examination_id)
                try:
                    _.initialize(self, calculate_smooth=True)
                except:
                    print(examination_id)
                    raise Exception("Could not initialize")
                _ = self.video_segmentation_prediction.find_one({"examination_id": examination_id})
                
                if as_object:
                    return VideoSegmentPrediction(**_)
                else:
                    return _

        except: 
            print("No prediction for {}".format(examination_id))
            return None

    def generate_examination_evaluator(self, examination_id):
        prediction = self.get_examination_segmentation_prediction(examination_id)
        if not prediction:
            pred = VideoSegmentPrediction(examination_id=examination_id)
            pred.initialize(self)
            prediction = self.get_examination_segmentation_prediction(examination_id)
        if prediction:
            prediction_id = prediction.id
        else: prediction_id = None
            
        annotation = self.get_examination_segmentation_annotation(examination_id)
        if annotation:
            annotation_id = annotation.id
        else:
            annotation_id = None
            
        return Evaluator(
            examination_id = examination_id,
            annotation_id = annotation_id,
            prediction_id = prediction_id,
            db = self
        )

    def get_examination_evaluator(self, examination_id, refresh = False):
        if refresh:
            _ = self.generate_examination_evaluator(examination_id)
            _.get_report_dictionary()
            self.evaluator.update_one(
                {"examination_id": examination_id},
                {"$set": _.to_dict()},
                upsert = True
                )
            _ = self.evaluator.find_one({"examination_id": examination_id})
            _["db"] = self
            _ = Evaluator(**_)
            
        else:
            _ = self.evaluator.find_one({"examination_id": examination_id})
            if _:
                _["db"] = self
                _ = Evaluator(**_)
            else:
                _ = self.generate_examination_evaluator(examination_id)
                if _.prediction_id:
                    _.get_report_dictionary()
                    self.evaluator.update_one(
                        {"examination_id": examination_id},
                        {"$set": _.to_dict()},
                        upsert = True
                        )
                    _ = self.evaluator.find_one({"examination_id": examination_id})
                    _["db"] = self
                    _ = Evaluator(**_)
        return _
    def get_multilabel_image_annotation(self, image_id: ObjectId):
        image = self.multilabel_annotation.find_one({"image_id": image_id})
        if image:
            return ImageAnnotations(**image)

        return {}

    def get_multilabel_image_annotations(self, image_ids: List[ObjectId]):
        images = self.multilabel_annotation.find({"image_id": {"$in": image_ids}})
        images = [ImageAnnotations(**image) for image in images]

        return images
    # def get_examination_annotations() # FIXME
    def get_examination_predictions(self, examination_id, as_object=True):
        r = self.multilabel_prediction.find({"examination_id": examination_id})
        if as_object:
            r = [Prediction(**p) for p in r]
        return r 

    def set_image_annotation(
        self, image_id: ObjectId, annotation: MultilabelAnnotation
    ):
        r = self.multilabel_annotation.update_one(
            {"_id": image_id},
            {"$set": {str(annotation.annotator_id): annotation.to_dict()}},
        )
        assert r.matched_count == 1
        return r

    def get_image_path(self, image_id):
        path = self.image.find_one({"_id": image_id})["path"]
        return Path(path)

    def get_examination_id_from_video_key(self, video_key: str):
        examination = self.examination.find_one({"video_key": video_key})
        if examination:
            return examination["_id"]
        else:
            warnings.warn(f"VIDEO KEY NOT FOUND {video_key}")
            return None

    def get_report_by_examination_id(self, examination_id: ObjectId):
        report = self.report.find_one({"examination_id": examination_id})
        if report:
            return Report(**report)

        return {}
    
    def get_report(self, report_id: ObjectId):
        report = self.report.find_one({"_id": report_id})
        if report:
            return Report(**report)

        return {}

    def sync_extern_examinations(self, url, auth):
        extern_examinations = get_extern_examinations(url, auth)
        existing_ids = self.examination.distinct("id_extern")
        new_extern_examinations = [e for e in extern_examinations if e.id_extern not in existing_ids]
        new_examinations = [Examination(**e.examination()) for e in new_extern_examinations]

        existing_video_keys = self.examination.distinct("video_key")
        new_examinations = [e for e in new_examinations if e.video_key not in existing_video_keys]

        for examination in tqdm(new_examinations):
            self.insert_examination(examination)
            

        return new_examinations

    def sync_extern_reports(self, url, auth):
        extern_examinations = get_extern_examinations(url, auth)
        existing_report_ids = self.report.distinct("id_extern")
        new_reports = [r.report() for r in extern_examinations if r.id_extern not in existing_report_ids]


        skip = []
        insert_reports = []
        for i, report in enumerate(new_reports):
            examination = self.examination.find_one({'id_extern': report['id_extern']})
            # assert examination
            if not examination:
                print("Skipped report with extern video id {}".format(report["id_extern"]))
                skip.append(i)
            else:
                report['examination_id'] = examination['_id']
                insert_reports.append(Report(**report))

        return insert_reports



    # def calculate_times(self):
    #     fps = self.fps

    # def get_report(self, _id: ObjectId, as_object = True):
    #     report = self.ASDASDASDASDASD.find_one({"examination_id": _id, "type": "report"})
    #     if as_object:
    #         report = ImageCollection(**report)

    #     return report
