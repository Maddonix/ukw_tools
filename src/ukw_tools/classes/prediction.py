from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any
from .base import PyObjectId
import pandas as pd
import numpy as np
from ..labels.utils import (
    predictions_to_array,
    calculate_smooth_predictions,
    get_intervention_segments,
    merge_nearby_segments,
    map_df,
    range_tuple_to_time,
    range_tuples_to_time
)
from ..plot.utils import segmentation_to_plot_df

WT_MAPPING = {
    "caecum": ["caecum", "ileocaecalvalve", "appendix", "ileum"],
    "intervention": ["nbi", "tool", "snare", "grasper", "needle", "clip", "polyp", "blood", "water_jet"],
    "outside": ["outside"]
}

wt_map_lookup = {}
for key, value in WT_MAPPING.items():
    for v in value:
        wt_map_lookup[v] = key

class Prediction(BaseModel):
    model_id: PyObjectId
    image_id: PyObjectId
    examination_id: PyObjectId
    prediction: List[float]
    prediction_smooth: Optional[List[float]]
    choices: List[str]
    labels: List[str]
    labels_smooth: Optional[List[str]]
    n: Optional[int]

class VideoSegmentPrediction(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    examination_id: PyObjectId
    wt_map_lookup: Dict[str, str] = wt_map_lookup
    fps: Optional[int]
    frame_count: Optional[int]

    prediction_df: Optional[pd.DataFrame]
    prediction_smooth_df: Optional[pd.DataFrame]
    prediction_wt_df: Optional[pd.DataFrame]
    prediction_segments: Optional[Dict[str, List[List[int]]]]
    prediction_smooth_segments: Optional[Dict[str, List[List[int]]]]
    prediction_wt_segments: Optional[Dict[str, List[List[int]]]]
    
    predicted_times: Optional[Dict[str, Optional[float]]]
    choices: Optional[List[str]]

    class Config:
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    def to_dict(self):
        _ = self.dict(include={
            "examination_id",
            "fps",
            "frame_count",
            "annotation_segments",
            "prediction_segments",
            "prediction_smooth_segments",
            "prediction_wt_segments",
            "predicted_times",
            "choices"
            }, exclude_none = True)
        _ = {k: v for k, v in _.items() if v is not None}
        return _

    def get_fps(self, db):
        examination = db.get_examination(self.examination_id)
        self.fps = examination.fps
        self.frame_count = examination.frame_count
        return examination.fps
    
    def get_prediction_df(self, db, smooth=False):
        preds = db.get_examination_predictions(self.examination_id)
        prediction_df, self.choices = predictions_to_array(preds, pred_smooth = smooth)
        return prediction_df

    def get_prediction_smooth_df(self, db, calculate_new = True):
        if calculate_new:
            examination = db.get_examination(self.examination_id)
            prediction_smooth_df = calculate_smooth_predictions(
                self.prediction_df,
                self.choices,
                int(examination.fps/2),
                future_frames = True,
                db = db,
                ignore_low_quality=True
            )
        else:
            prediction_smooth_df = self.get_prediction_df(db, smooth=True)
        return prediction_smooth_df

    def map_df(self, df, wt_map_lookup=None):
        if not wt_map_lookup:
            wt_map_lookup = self.wt_map_lookup
        df = map_df(df, wt_map_lookup)
        return df

    def get_segments(self, df, labels, max_diff= None, min_length = None):
        if not min_length:
            min_length = self.fps*1
        if not max_diff:
            max_diff = self.fps * 5
        segments = get_intervention_segments(df, labels, min_length)
        for key, value in segments.items():
            if key == "low_quality": 
                continue
            if key == "intervention":
                max_diff = self.fps*30
            segments[key] = merge_nearby_segments(value, max_diff)

        # if "caecum" in labels:
        #     _max = self.frame_count
        #     if 

        return segments

    def post_process_wt_segments(self, segments):
        new = {}

        if not segments["caecum"]:
            print("No caecum segmentation")
            return None
        start = 0
        stop = self.frame_count
        if segments["outside"]:
            _start = segments["outside"][0][1]
            if _start < self.frame_count/2:
                start=_start
            _stop = segments["outside"][-1][0]
            if _stop > self.frame_count/2:
                stop = _stop
        
        new["insertion"] = [(start, segments["caecum"][0][0])]
        
        _caecum = [(segments["caecum"][0][0], segments["caecum"][-1][1])]
        # sanity check
        print(_caecum)
        _len = len(segments["caecum"])
        i = _len - 2
        _wt_tuple = [_caecum[0][1], stop]
        _t = range_tuple_to_time(_wt_tuple, self.fps)
        print(_t)
        while i >= 0 and _t<100:
            print("RECALC")
            _caecum = [(segments["caecum"][0][0], segments["caecum"][i][1])]
            _wt_tuple = [_caecum[0][1], stop]
            _t = range_tuple_to_time(_wt_tuple, self.fps)
            i -= 1
            print(_caecum, _t)


        new["caecum"] = _caecum
        new["intervention"] = segments["intervention"]
        new["withdrawal"] = [(new["caecum"][-1][1], stop)]

        return new

    def get_plot_df(self, segmentation):
        return segmentation_to_plot_df(
            segmentation, self.frame_count
        )

    def calculate_times(self, segments):
        times = {}
        categories = ["insertion", "caecum", "withdrawal"]
        for category in categories:
            times[category] = range_tuples_to_time(segments[category], self.fps)

        interventions = {cat:0 for cat in categories}

        for intervention_range in segments["intervention"]:
            start = intervention_range[0]
            if start < segments["caecum"][0][0]:
                interventions["insertion"] += range_tuple_to_time(intervention_range, self.fps)
            if start < segments["withdrawal"][0][0]:
                interventions["caecum"] += range_tuple_to_time(intervention_range, self.fps)
            if start >= segments["withdrawal"][0][0]:
                interventions["withdrawal"] += range_tuple_to_time(intervention_range, self.fps)

        for category in categories:
            times[f"{category}_corrected"] = times[category] - interventions[category]

        return times


    def initialize(self, db, calculate_smooth=True):
        self.get_fps(db)
        # Implement get annotations

        self.prediction_df = self.get_prediction_df(db)
        self.prediction_smooth_df = self.get_prediction_smooth_df(db, calculate_smooth)
        self.prediction_wt_df = map_df(self.prediction_smooth_df, self.wt_map_lookup)

        self.prediction_smooth_segments = self.get_segments(
            self.prediction_smooth_df,
            self.choices,
            max_diff=self.fps*10,
            min_length=self.fps*1.5
        )

        self.prediction_wt_segments = self.get_segments(
            self.prediction_wt_df,
            list(WT_MAPPING.keys()),
            max_diff=self.fps*10,
            min_length=self.fps*1.5
        )
        self.prediction_wt_segments = self.post_process_wt_segments(self.prediction_wt_segments)
        if self.prediction_wt_segments:
            self.predicted_times = self.calculate_times(self.prediction_wt_segments)
        else:
            categories = ["insertion", "caecum", "withdrawal"]
            self.predicted_times = {cat:None for cat in categories}
            for cat in categories:
                self.predicted_times[f"{cat}_corrected"] = None

        db.video_segmentation_prediction.update_one({
            "examination_id": self.examination_id
        },{
            "$set": self.to_dict()
        }, upsert = True
        )


