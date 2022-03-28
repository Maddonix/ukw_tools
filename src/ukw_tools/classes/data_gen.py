from faker import Faker
from .examination import Examination
from pathlib import Path

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


