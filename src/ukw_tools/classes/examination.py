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
from datetime import datetime as dt
from ..media.video import get_video_info

class ExaminationDashboardData(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    examination_id: PyObjectId
    age: Optional[int]
    gender: Optional[int]
    video_key: Optional[str]
    fully_extracted: Optional[bool]
    fps: Optional[int]
    is_video: Optional[bool]
    frame_count: Optional[int]
    origin_category: Optional[str]
    crop: Optional[Tuple[int, int, int, int]]
    cecum_reached: Optional[bool]
    has_video_segmentation_prediction: Optional[bool]

    has_examination_report: Optional[bool]
    has_histo_report: Optional[bool]
    has_report_annotation: Optional[bool]

    n_annotated_images: Optional[int]
    n_freezes_detected: Optional[int]
    examiner: Optional[str]
    examination_type: Optional[str]

    ai_vision_path: Optional[Path]
    
    n_detected_polyp_sequences: Optional[int] # To Do
    ileum_id: Optional[PyObjectId]
    appendix_id: Optional[PyObjectId]
    ileocaecalvalve_id: Optional[PyObjectId]
    # n_polypectomy_sequences: Optional[int] # To Do

    def to_db(self, db):
        _dict = self.dict()
        _dict.pop("id")
        db.examination_dashboard_data.update_one(
            {"examination_id": self.examination_id},
            {"$set": _dict},
            upsert=True
        )

    def refresh(self, db, upload = True):
        self.refresh_examination_data(db)
        self.refresh_is_extracted(db)
        self.refresh_report(db)
        self.refresh_freezes(db)
        self.refresh_evaluator(db)
        self.refresh_multilabel_annotations(db)
        if self.is_video:
            eval = db.get_examination_evaluator(self.examination_id)
            eval.get_elements()
            self.refresh_has_video_segmentation_prediction(db)
        if upload:
            self.to_db(db)

    def refresh_has_video_segmentation_prediction(self, db):
        segm_pred = db.get_examination_segmentation_prediction(self.examination_id)
        if segm_pred:
            self.has_video_segmentation_prediction = True
            try:
                self.n_detected_polyp_sequences = len(segm_pred.prediction_smooth_segments["polyp"])
                self.cecum_reached = len(segm_pred.prediction_wt_segments["caecum"])>0
            except:
                print(segm_pred)
                raise Exception
        else:
            self.has_video_segmentation_prediction = False
            self.n_detected_polyp_sequences = None
            self.cecum_reached = None


    def refresh_examination_data(self, db):
        exam = db.get_examination(self.examination_id)
        if hasattr(exam, "video_key"):
            self.video_key = exam.video_key
        if hasattr(exam, "age"):
            if isinstance(exam.age, int):
                if exam.age > 0:
                    self.age = exam.age
            
        if hasattr(exam, "gender"):
            if isinstance(exam.gender, int):
                assert exam.gender in [0,1,2]
                self.gender = exam.gender

        if hasattr(exam, "crop"):
            if exam.crop:
                self.crop = exam.crop

        if hasattr(exam, "examiners"):
            if isinstance(exam.examiners, list):
                self.examiner = exam.examiners[0]

        self.origin_category = exam.origin_category
        self.is_video = exam.is_video
        if self.is_video:
            self.fps = exam.fps
            self.frame_count = exam.frame_count

        self.examination_type = exam.examination_type

    def refresh_is_extracted(self, db):
        # Is Extracted?
        _ = db.image.find_one({"examination_id": self.examination_id, "is_extracted": False})
        if _: self.fully_extracted = False
        else: self.fully_extracted = True

    def refresh_report(self, db):
        # Report
        r = db.get_report_by_examination_id(self.examination_id)
        if r and hasattr(r, "examination"):
            if r.examination: self.has_examination_report=True
            else: self.has_examination_report=False
        else: self.has_examination_report=False
        if r and hasattr(r, "histo"):
            if r.histo: self.has_histo_report=True
            else: self.has_histo_report=False
        else: self.has_histo_report=False
        if r and hasattr(r, "report_annotation"):
            if r.report_annotation: self.has_report_annotation=True
            else: self.has_report_annotation=False
        else: self.has_report_annotation=False

    def refresh_evaluator(self, db):
        try:
            evaluator = db.get_examination_evaluator(self.examination_id)
            self.n_detected_polyp_sequences = len(evaluator.report["polyps"])

            if "ileum" in evaluator.report["id_landmarks"]:
                self.ileum_id = evaluator.report["id_landmarks"]["ileum"]
            if "appendix" in evaluator.report["id_landmarks"]:
                self.appendix_id = evaluator.report["id_landmarks"]["appendix"]
            if "ileocaecalvalve" in evaluator.report["id_landmarks"]:
                self.ileocaecalvalve_id = evaluator.report["id_landmarks"]["ileocaecalvalve"]
        except:
            pass

    def refresh_freezes(self, db):
        freezes = db.freeze_detection.find_one({"examination_id": self.examination_id})
        if freezes:
            self.n_freezes_detected = len(freezes["freezes"])

    def refresh_multilabel_annotations(self, db):
        n_annotations = db.multilabel_annotation.count_documents({"examination_id": self.examination_id})
        self.n_annotated_images = n_annotations

    
class Examination(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    origin: str
    origin_category: str
    examination_type: str
    is_video: bool
    video_key: Optional[str]
    id_extern: Optional[int]
    examiners: Optional[List[str]]
    date: Optional[dt]
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