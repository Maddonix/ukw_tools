from datetime import datetime as dt
from pathlib import Path

from faker import Faker

from .annotation import MultilabelAnnotation
from .examination import Examination
from .report import ReportAnnotationResult, ReportPolypAnnotationResult


class DataGen:
    def __init__(self, data_dir: Path):
        self.factory = Faker()
        self.dir = data_dir

    def text(self):
        return self.factory.text()

    def examination_with_video(self):
        examination = Examination(
            origin = "test",
            origin_category = "test",
            is_video = True,
            path = self.dir.joinpath('test_video.mp4'),
            examination_type = "unknown",
        )
        assert examination.path.exists()
        return examination

    def multilabel_annotation(self):
        annotation = {
            "source": "test",
            "annotator_id": 999,
            "date": dt.now(),
            "name": "test",
            "value": 1,
            "choices": ["test1", "test2", "test3"],
        }

        return MultilabelAnnotation(**annotation)

    def polyp_report_annotation(self):
        report = {
            "location_segment": "sigma",
            "location_cm": 16,
            "size_category": "<5mm",
            "resection": False
        }

        report = ReportPolypAnnotationResult(**report)
        return report

    def report_annotation(self):
        report = {
            "polyps": [self.polyp_report_annotation()],
            "withdrawal_time": 300, # in s
            "n_polyps": 1,
            "indication": "screening"
        }

        return ReportAnnotationResult(**report)




