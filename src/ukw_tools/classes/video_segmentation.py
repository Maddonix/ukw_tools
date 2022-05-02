from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from .annotation import Flank
from .base import PyObjectId
import pandas as pd
from ..labels.utils import (
    map_df,
    get_intervention_segments,
    range_tuple_to_time,
    range_tuples_to_time
)
from ..plot.utils import segmentation_to_plot_df

WT_MAPPING = {
    "caecum": ["caecum", "ileocaecalvalve", "appendix", "ileum"],
    "intervention": ["nbi", "tool", "snare", "grasper", "needle", "clip", "polyp", "blood", "water_jet", "resection"],
    "outside": ["outside"]
}

wt_map_lookup = {}
for key, value in WT_MAPPING.items():
    for v in value:
        wt_map_lookup[v] = key

class VideoSegmentation(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    fps: Optional[int]
    frame_count: Optional[int]
    wt_map_lookup: Dict[str, str] = wt_map_lookup
    annotation_df: Optional[pd.DataFrame] # Optional[Any] 
    annotation_segments:Optional[Dict[str, List[List[int]]]]
    annotation_wt_segments:Optional[Dict[str, List[List[int]]]]
    examination_id: Optional[PyObjectId]
    annotated_times: Optional[Dict[str, float]]
    wt_freeze_0: Optional[float]
    wt_freeze_1: Optional[float]

    class Config:
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    def to_dict(self):
        _ = self.dict(exclude_none=True, exclude_defaults=True, exclude = {"annotation_df"})
        return _

    def map_df(self, df, wt_map_lookup=None):
        if not wt_map_lookup:
            wt_map_lookup = self.wt_map_lookup
        df = map_df(df, wt_map_lookup)
        return df

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

        # Caecum sanity check:
        # c_start_index = 0
        # c_stop_index = 0
        # for i, c in enumerate(segments["caecum"]):
            
        
        new["insertion"] = [(start, segments["caecum"][0][0])]
        new["caecum"] = [(segments["caecum"][0][0], segments["caecum"][-1][1])]
        new["intervention"] = segments["intervention"]
        new["withdrawal"] = [(segments["caecum"][-1][1], stop)]

        return new
    def get_plot_df(self, segmentation):
        return segmentation_to_plot_df(
            segmentation, self.frame_count
        )
        
    def get_wt_segments(self):
        plot_df = segmentation_to_plot_df(self.annotation_segments, self.frame_count)
        plot_df_wide = plot_df.pivot(index = "n", columns = "label", values = "value").fillna(0).reset_index()
        mapped_df = self.map_df(plot_df_wide)
        LABELS = ["caecum", "intervention", "outside"]
        segments = get_intervention_segments(mapped_df, LABELS, self.fps)
        segments = self.post_process_wt_segments(segments)

        return segments

    def calculate_wt_from_freezes(self):
        freezes = self.annotation_segments["freeze"]
        caecum = [
            self.annotation_segments["caecum"][0][0],
            self.annotation_segments["caecum"][-1][1]
        ]
        freezes = [_ for _ in freezes if _[0] > caecum[0] and _[0] < caecum[1]]

        if not freezes:
            return None, None

        if not self.annotation_segments["outside"]:
            wt_end = self.frame_count
        else:
            wt_end = self.annotation_segments["outside"][-1][0]
            if wt_end < caecum[1]:
                wt_end = self.frame_count

        wt_0 = range_tuple_to_time([freezes[0][0], wt_end], self.fps)
        wt_1 = range_tuple_to_time([freezes[-1][0], wt_end], self.fps)

        self.wt_freeze_0 = wt_0
        self.wt_freeze_1 = wt_1

        return wt_0, wt_1

    def calculate_times(self, segments):
        times = {}
        categories = ["insertion", "caecum", "withdrawal"]
        for category in categories:
            times[category] = range_tuples_to_time(segments[category], self.fps)
        interventions = {cat:0 for cat in categories}

 
        for intervention_range in segments["intervention"]:
            
            start = intervention_range[0]
            stop = intervention_range[1]

            if start < segments["caecum"][0][0]:
                if stop <= segments["caecum"][0][0]:
                    interventions["insertion"] += range_tuple_to_time(intervention_range, self.fps)
                else:
                    interventions["insertion"] += range_tuple_to_time([start, segments["caecum"][0][0]], self.fps)
                    interventions["caecum"] += range_tuple_to_time([segments["caecum"][0][0], stop], self.fps)
            elif start < segments["withdrawal"][0][0]:
                if stop <= segments["withdrawal"][0][0]:
                    interventions["caecum"] += range_tuple_to_time(intervention_range, self.fps)
                else:
                    interventions["caecum"] += range_tuple_to_time([start, segments["withdrawal"][0][0]], self.fps)
                    interventions["withdrawal"] += range_tuple_to_time([segments["withdrawal"][0][0], stop], self.fps)
            elif start >= segments["withdrawal"][0][0]:
                interventions["withdrawal"] += range_tuple_to_time(intervention_range, self.fps)

        for category in categories:
            times[f"{category}_corrected"] = times[category] - interventions[category]

        return times

    # def get_fps(self, db):
    #     examination = db.get_examination(self.examination_id)
    #     self.fps = examination.fps
    #     self.frame_count = examination.frame_count
    #     return examination.fps

    
