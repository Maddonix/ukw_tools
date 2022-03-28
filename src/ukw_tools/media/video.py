from typing import Tuple

import cv2


def get_video_info(video_path: str) -> Tuple:
    """
    Get video info from a video file.
    
    Returns fps, frame_count
    """

    cap = cv2.VideoCapture(video_path.as_posix())
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    return int(fps), int(frame_count)
