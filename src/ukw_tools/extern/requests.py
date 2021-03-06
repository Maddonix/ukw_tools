import requests
from urllib3.exceptions import InsecureRequestWarning
from .classes import ExternAnnotatedVideo, ExternVideoFlankAnnotation, VideoExtern
from .smartie import process_smartie_record
import pandas as pd

requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

def get_extern_annotations(url, auth):
    r = requests.get(f"{url}/GetVideosWithAnnotations", auth=auth, verify=False)
    assert r.status_code == 200
    r = [ExternAnnotatedVideo(**_) for _ in r.json()]

    return r

def get_extern_video_annotation(video_key, url, auth):
    r = requests.get(url+"/GetAnnotationsByVideoName/"+video_key, auth=auth, verify=False)
    assert r.status_code == 200
    annotations = [ExternVideoFlankAnnotation(**_) for _ in r.json()]

    return annotations

def get_smartie_data(url, auth, as_df = True):
    r = requests.get(url+"/GetSmartieVideoData", auth = auth, verify=False)
    assert r.status_code == 200

    r = r.json()
    if as_df:
        r = [process_smartie_record(_) for _ in r]
        r = pd.DataFrame.from_records(r)
    return r

def get_extern_examinations(url, auth):
    r = requests.get(f"{url}/GetVideosExtern", auth=auth, verify=False)
    assert r.status_code == 200
    videos = [VideoExtern(**_) for _ in r.json()]

    return videos
