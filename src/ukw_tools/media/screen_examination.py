from pathlib import Path


from bson import ObjectId
import json
from ukw_tools.model.multilabel_classification_net import MultilabelClassificationNet
from ukw_tools.dataset.image_classification import ImageClassificationDs
from ukw_tools.classes.prediction import VideoSegmentPrediction
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import os

from ..classes.db import DbHandler
from .video import extract_frame_list


def extract_video(payload):
    """Extracts all Frames of an examination

    Args:
        payload {"id": i, "examination": examination, "mongo_url": mongo_url}
    """
    db = DbHandler(payload["mongo_url"])
    examination = payload["examination"]
    print(examination.video_key)
    frame_dir = Path("/extreme_storage/files/frames/").joinpath(examination.video_key)
    if not frame_dir.exists():
        print("DIR DOESNT EXIST, CREATING FOLDER")
        os.mkdir(frame_dir)

    image_collection = db.get_examination_image_collection(examination.id)
    missing = []
    id_not_extracted = []
    n_not_extracted = []

    ids = list(image_collection.images.values())
    images = db.get_images(ids)
    if not len(images) == len(ids):
        _ids = [_.id for _ in images]
        for _id in _ids:
            if _id not in ids:
                missing.append(_id)

    for image in tqdm(images):
        if not image.exists():
            id_not_extracted.append(image.id)
            n_not_extracted.append(image.n)

    path_dict = extract_frame_list(examination.path, n_not_extracted, frame_dir)
    db.update_frames_extracted(examination.id, path_dict)


def get_examination_dataloader(
    examination_id, db, scaling = 69, batch_size=12,
    num_workers=4, shuffle=False, training = False):
    if training:
        predict=False
    else:
        predict=True
    
    image_collection = db.get_examination_image_collection(examination_id)
    paths, labels, crop, choices = db.prepare_ds_from_image_collection(image_collection.id, predict=predict)
    ds = ImageClassificationDs(paths, labels, crop, scaling=scaling, training = training)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    return dl

def predict_batch(x, model, cuda=False):
    if cuda:
        x = x.cuda()
    else:
        x = x.to("cpu")
    with torch.no_grad():
        pred = model(x)
    pred = pred.detach().cpu().numpy()
    return pred

def prediction_to_record(pred, _y, target_labels):
    labels = [target_labels[i] for i in np.where(pred>0.5)[0]]

    record = {
        "image_id": _y,
        "labels": labels,
        "prediction": pred.tolist(),
        "choices": target_labels
    }

    return record

def predict_examination(payload):
    db = DbHandler(payload["mongo_url"])
    examination = payload["examination"]
    model_id = payload["model_id"]
    upload = payload["upload"]
    cuda = payload["cuda"]
    print(examination.video_key)
    checkpoint_path = db.model.find_one({"_id": model_id})["trainer"]["model_path"]
    model = MultilabelClassificationNet.load_from_checkpoint(checkpoint_path)
    model.cuda()
    model.eval()
    target_labels = model.labels

    records = []   
    target_labels = model.labels
    dl = get_examination_dataloader(
        examination.id, db, scaling = 69, batch_size = 12,
        num_workers = 4, shuffle = False, training = False
        )

    for batch in tqdm(dl):
        x, y = batch
        pred = predict_batch(x, model, cuda)
        y = [ObjectId(_) for _ in y]
        if upload:
            frame_number_lookup = {_["_id"]: _["n"] for _ in db.image.find({"_id": {"$in": y}})}
        for i,v in enumerate(pred):
            record = prediction_to_record(v, y[i], target_labels)
            record["examination_id"] = examination.id
            record["n"] = frame_number_lookup[y[i]]
            if upload:
                db.db.MultilabelPrediction.update_one({"image_id": y[i]}, {"$set": record}, upsert=True)
            
            record["image_id"] = str(record["image_id"])
            record["examination_id"] = str(examination.id)
            records.append(record)


    with open(f"results/predictions_{examination.video_key}.json", "w") as f:
        json.dump(records, f)
    
    try:
        prediction = VideoSegmentPrediction(examination_id= examination.id)
        prediction.initialize(db)
    except:
        pass