from typing import Tuple, List
from pathlib import Path
import cv2


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
