from ukw_tools.classes.data_gen import DataGen
from pathlib import Path


factory = DataGen(Path("tests/test_data"))

def test_generate_examination_with_video():
    examination = factory.examination_with_video()
    assert examination.is_video == True
    assert examination.examination_type == "unknown"

def test_get_video_info():
    examination = factory.examination_with_video()
    fps, frame_count = examination.get_video_info()
    assert fps == 25
    assert frame_count == 360

def test_generate_frame_list():
    examination = factory.examination_with_video()
    frames = examination.generate_frame_list()
    fps, frame_count = examination.get_video_info()
    assert len(frames) == frame_count