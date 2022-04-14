from typing import Any, Optional, Dict, Union, List
from bson import ObjectId
from pathlib import Path
import pathlib
import pydantic.json
from pydantic import BaseModel, Field
from ukw_tools.classes.base import PyObjectId
from ukw_tools.classes.prediction import VideoSegmentPrediction
from ukw_tools.classes.video_segmentation import VideoSegmentation
from ukw_tools.labels.utils import predictions_to_array
from ..plot.utils import get_plot

# class EvaluatorReport(BaseModel):
pydantic.json.ENCODERS_BY_TYPE[pathlib.PosixPath] = str
pydantic.json.ENCODERS_BY_TYPE[pathlib.WindowsPath] = str
pydantic.json.ENCODERS_BY_TYPE[Path] = str

class Evaluator(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    examination_id: Optional[PyObjectId]
    annotation_id: Optional[PyObjectId]
    prediction_id: Optional[PyObjectId]

    db: Optional[Any]
    annotation: Optional[VideoSegmentation]
    prediction: Optional[VideoSegmentPrediction]

    prediction_df: Optional[Any]
    report: Optional[
        Dict[
            str,
            Any
            # Union[
            #     ObjectId,
            #     Path,
            #     str,
            #     Dict[str, Union[Path, ObjectId, str, float]],
            #     List[Dict[str, Union[Path, ObjectId, str, float]]],
            #     None
            # ]
        ]
    ]

    def get_elements(self):
        self.prediction = self.db.get_examination_segmentation_prediction(
            self.examination_id
        )
        self.annotation = self.db.get_examination_segmentation_annotation(
            self.examination_id
        )

    class Config:
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    def to_dict(self):
        new = self.dict(
            exclude_none=True,
            include={"id", "examination_id", "annotation_id", "prediction_id", "report"},
        )
        _ = new["report"]

        if "polyps" in _:
            if _["polyps"]:
                for i in range(len(_["polyps"])):
                    for key, value in _["polyps"][i].items():
                        if isinstance(value, type(Path("."))):
                            _["polyps"][i][key] = str(value)

        if "landmarks" in _:
            if _["landmarks"]:
                for key, value in _["landmarks"].items():
                    if isinstance(value, type(Path("."))):
                        _["landmarks"][key] = str(value)        
        new["report"] = _

        return new

    def get_times(self):
        self.get_elements()
        a = None
        p = None
        if self.annotation:
            if not hasattr(self.annotation, "annotation_segments"):
                a = None
            else:
                if not hasattr(self.annotation, "annotation_wt_segments"):
                    self.annotation.annotation_wt_segments = (
                        self.annotation.get_wt_segments(
                            self.annotation.frame_count, self.annotation.fps
                        )
                    )
                if not hasattr(self.annotation, "annotated_times"):
                    self.annotation.annotated_times = self.annotation.calculate_times(
                        self.annotation.annotation_wt_segments
                    )
                self.db.video_segmentation.update_one(
                    {"examination_id": self.examination_id},
                    {"$set": self.annotation.to_dict()},
                )
                a = self.annotation.annotated_times
        if self.prediction:
            if not hasattr(self.prediction, "predicted_times"):
                self.prediction.initialize(self.db)
            if not hasattr(self.prediction, "predicted_times"):
                p = None
            else:
                p = self.prediction.predicted_times

        times = {"annotation": a, "prediction": p}
        return times

    def get_landmark_image_dict(self):
        landmarks = ["appendix", "ileocaecalvalve", "ileum"]
        if isinstance(self.prediction_df, type(None)):
            predictions = self.db.get_examination_predictions(self.examination_id)
            self.prediction_df, choices = predictions_to_array(
                predictions, pred_smooth=False, raw=True
            )

        landmark_images = {}
        for landmark in landmarks:
            _segments = self.prediction.prediction_smooth_segments[landmark]
            _frame_list = []
            for segment in _segments:

                _frame_list.extend(range(segment[0], segment[1]))
            _selection = self.prediction_df.loc[
                self.prediction_df.n_frame.isin(_frame_list)
            ]
            _selection = _selection.loc[
                (_selection[landmark] > 0.5) & (_selection["low_quality"] < 0.5),
                ["image_id", landmark],
            ]
            if len(_selection):
                idx_max = _selection[landmark].idxmax()
                _image_id = _selection.loc[idx_max].image_id
                landmark_images[landmark] = _image_id
        return landmark_images

    def get_polyp_sequences(self):
        if isinstance(self.prediction_df, type(None)):
            predictions = self.db.get_examination_predictions(self.examination_id)
            self.prediction_df, choices = predictions_to_array(
                predictions, pred_smooth=False, raw=True
            )

        segments = self.prediction.prediction_smooth_segments

        polyp_sequence_template = {
            "start": None,
            "stop": None,
            "image_id": None,
            "instrument": None,
            "image_id_instrument": None,
            "nbi": None,
            "image_id_nbi": None,
        }

        polyp_flanks = segments["polyp"]

        polyp_sequences = []
        for p_flank in polyp_flanks:
            polyp_sequence = polyp_sequence_template.copy()
            polyp_sequence["start"] = p_flank[0]
            polyp_sequence["stop"] = p_flank[1]

            frame_list = list(range(p_flank[0], p_flank[1]))
            if isinstance(self.prediction_df, type(None)):
                predictions = self.db.get_examination_predictions(self.examination_id)
                self.prediction_df, choices = predictions_to_array(
                    predictions, pred_smooth=False, raw=True
                )

            p_df = self.prediction_df.loc[self.prediction_df.n_frame.isin(frame_list)]
            try:
                select = p_df.loc[(p_df["nbi"] < 0.5) & (p_df["low_quality"] < 0.5)]
                idx_max = select["polyp"].idxmax()
                polyp_sequence["image_id"] = select.loc[idx_max].image_id
            except:
                idx_max = p_df["polyp"].idxmax()
                polyp_sequence["image_id"] = p_df.loc[idx_max].image_id

            # get tools

            g_selection = p_df.loc[
                (p_df["grasper"] > 0.5) & (p_df["low_quality"] < 0.5)
            ]
            s_selection = p_df.loc[(p_df["snare"] > 0.5) & (p_df["low_quality"] < 0.5)]
            if len(g_selection) > len(s_selection):
                selection = g_selection
                tool_type = "grasper"
            else:
                selection = s_selection
                tool_type = "snare"

            if len(selection):
                idx_max = selection[tool_type].idxmax()
                polyp_sequence["instrument"] = tool_type
                polyp_sequence["image_id_instrument"] = selection.loc[idx_max].image_id
            else:
                polyp_sequence["instrument"] = None
                polyp_sequence["image_id_instrument"] = None

            # get nbi
            selection = p_df.loc[(p_df["nbi"] > 0.5) & (p_df["low_quality"] < 0.5)]
            if len(selection):
                try:
                    idx_max = selection["polyp"].idxmax()
                except:
                    idx_max = selection["nbi"].idxmax()
                polyp_sequence["nbi"] = True
                polyp_sequence["image_id_nbi"] = selection.loc[idx_max].image_id
            else:
                polyp_sequence["nbi"] = False
                polyp_sequence["image_id_nbi"] = None

            polyp_sequences.append(polyp_sequence)

        return polyp_sequences

    def get_report_dictionary(self):
        self.get_elements()
        times = self.get_times()
        predicted_times = times["prediction"]
        polyps = self.get_polyp_sequences()
        for i, polyp_sequence in enumerate(polyps):
            if polyp_sequence["image_id"]:
                polyps[i]["image_path"] = self.db.get_image_path(
                    polyp_sequence["image_id"]
                )
            if polyp_sequence["image_id_instrument"]:
                polyps[i]["image_path_instrument"] = self.db.get_image_path(
                    polyp_sequence["image_id_instrument"]
                )
            if polyp_sequence["image_id_nbi"]:
                polyps[i]["image_path_nbi"] = self.db.get_image_path(
                    polyp_sequence["image_id_nbi"]
                )

        landmarks = self.get_landmark_image_dict()
        landmarks = {
            key: self.db.get_image_path(value)
            for key, value in landmarks.items()
            if value
        }

        report = {}
        try:
            report["predicted_times"] = predicted_times
        except:
            report["predicted_times"] = None
        try:
            report["annotated_times"] = self.annotation.annotated_times
        except:
            report["annotated_times"] = None
        report["polyps"] = polyps
        report["landmarks"] = landmarks
        self.report = report
        return report

    def pred_plot(self, plot_type, x_as_time = False):
        if plot_type == "withdrawal_time":
            _segmentation = self.prediction.prediction_wt_segments
        elif plot_type=="detail":
            _segmentation = self.prediction.prediction_smooth_segments
        plot_df = self.prediction.get_plot_df(_segmentation)
        if x_as_time:
            fps = self.prediction.fps
        else: fps = None
        plot = plot=get_plot(plot_df, fps)

        return plot

    def annotation_plot(self, x_as_time = False):
        _segmentation = self.annotation.annotation_wt_segments

        plot_df = self.annotation.get_plot_df(_segmentation)
        if x_as_time:
            fps = self.annotation.fps
        else: fps = None
        plot = plot=get_plot(plot_df, fps)

        return plot
        
