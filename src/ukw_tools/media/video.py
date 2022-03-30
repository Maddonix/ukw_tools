from typing import Tuple, List
from pathlib import Path
import cv2
import warnings


def get_frame_name(frame_index: int) -> str:
    """
    Get frame name from frame index.
    """

    return f"{frame_index:07d}.jpg"


def get_video_info(video_path: str) -> Tuple:
    """
    Get video info from a video file.

    Returns fps, frame_count
    """

    cap = cv2.VideoCapture(video_path.as_posix())
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if frame_count < 1:
        warnings.warn("Frame count is less than 1. Trying to get frame count from video file.")
        frame_count = 0
        ret = True
        while ret:
            ret, frame = cap.read()
            frame_count += 1
        
    return int(fps), int(frame_count)


def extract_frame_list(
    video_path: Path, frame_list: List[int], frame_dir: Path
) -> List:
    """
    Extract frame list from a video file.
    """

    cap = cv2.VideoCapture(video_path.as_posix())
    saved = {}
    for i in frame_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            path = frame_dir.joinpath(get_frame_name(i))
            cv2.imwrite(path.as_posix(), frame)
            saved[i] = path

    return saved
