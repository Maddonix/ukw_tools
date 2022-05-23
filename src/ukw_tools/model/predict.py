from .multilabel_classification_net import MultilabelClassificationNet
from ..dataset.image_classification import ImageClassificationDs
from ..classes.prediction import VideoSegmentPrediction
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from bson import ObjectId
import numpy as np
from pathlib import Path
import os
import json

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

def predict_examination(examination, model, model_id, db=None, cuda = True, upload = False, out = Path("./preds")):
    print(examination.video_key)
    if upload == False:
        if not out.exists():
            os.mkdir(out)
    records = []   
    target_labels = model.labels
    dl = get_examination_dataloader(
        examination.id, db, scaling = 70, batch_size = 12,
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
            record["model_id"] = model_id
            if upload:
                db.db.MultilabelPrediction.update_one({"image_id": y[i]}, {"$set": record}, upsert=True)
            
            record["image_id"] = str(record["image_id"])
            record["examination_id"] = str(examination.id)
            records.append(record)


    if not upload:
        with open(out.joinpath(f"predictions_{examination.video_key}.json"), "w") as f:
            json.dump(records, f)
    
    try:
        prediction = VideoSegmentPrediction(examination_id= examination.id)
        prediction.initialize(db)
    except:
        print("to initialize predictions!")
