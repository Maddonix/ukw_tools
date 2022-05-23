import warnings
from pathlib import Path
from typing import List, Tuple
import os
import cv2
from tqdm import tqdm

def get_delta(examination, db, dim):
    image_collection = db.get_examination_image_collection(examination.id)
    crop = examination.crop
    center = (
        int((crop[1] - crop[0])/2),
        int((crop[3] - crop[2])/2)
    )
    crop=(center[0]-dim, center[0]+dim, center[1]-dim, center[1]+dim)

    evaluator = db.get_examination_evaluator(examination.id)
    evaluator.get_elements()
    exclude = []
    if evaluator.annotation: 
        if "outside" in evaluator.annotation.annotation_segments:
            for start, stop in evaluator.annotation.annotation_segments["outside"]:
                exclude.extend([_ for _ in range(start, stop)])

    image_ids = list(image_collection.images[_] for _ in image_collection.images.keys())
    images = db.get_images(image_ids)
    
    delta = []
    for i,image in tqdm(enumerate(images)):       
        if i == 0:
            img = cv2.imread(image.path.as_posix())[crop[0]:crop[1], crop[2]:crop[3]]
            delta.append(10e9)
        elif i in exclude:
            delta.append(10e9)
        else:
            img_p = img
            img = cv2.imread(image.path.as_posix())[crop[0]:crop[1], crop[2]:crop[3]]
            _diff = img - img_p

            delta.append(_diff.sum())

    return delta

def get_frame_dir(video_key, base_dir = Path("/extreme_storage/files/frames")):
    frame_dir = base_dir.joinpath(video_key)
    if not frame_dir.exists():
        os.mkdir(frame_dir)
    return frame_dir

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
    video_path: Path, frame_list: List[int], frame_dir: Path, all = False
) -> List:
    """
    Extract frame list from a video file.
    """
    frame_list.sort()
    cap = cv2.VideoCapture(video_path.as_posix())
    saved = {}
    if all:
        for i in tqdm(frame_list):
            ret, frame = cap.read()
            if ret:
                path = frame_dir.joinpath(get_frame_name(i))
                cv2.imwrite(path.as_posix(), frame)
                saved[i] = path
    else:
        for i in tqdm(frame_list):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                path = frame_dir.joinpath(get_frame_name(i))
                cv2.imwrite(path.as_posix(), frame)
                saved[i] = path

    return saved
    