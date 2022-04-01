from itertools import groupby
from operator import itemgetter
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import pandas as pd

def get_consecutive_ranges(list) -> List[Tuple]:
    range_tuples = []
    for k, g in groupby(enumerate(list), lambda x: x[0]-x[1]):
        range_tuples.append([_ for _ in map(itemgetter(1), g)])

    for i in range(len(range_tuples)):
        range_tuples[i] = (range_tuples[i][0], range_tuples[i][-1]+1)

    return range_tuples

def running_mean(preds, conv_len, weight_main=None, threshold=0.5, future_frames=True):
    if not conv_len % 2:
        conv_len += 1

    if not weight_main:
        weight_main = 1/conv_len

    weight_main = conv_len * weight_main

    if future_frames:
        conv = np.ones(int(conv_len))

    else:
        conv = np.zeros(int(conv_len-1))
        conv = np.append(conv, np.ones(conv_len))

    mid = int(conv_len/2)+1
    conv[mid] = weight_main
    conv = conv/sum(conv)

    smooth_prediction = np.convolve(preds, conv, mode="same")
    smooth_prediction[smooth_prediction > threshold] = 1
    smooth_prediction[smooth_prediction <= threshold] = 0
    return smooth_prediction

def calculate_smooth_predictions(df, choices, conv_len=25, future_frames=True, db = None, ignore_low_quality=False):
    df_smooth = df.copy()

    for choice in choices:
        df_smooth[choice] = running_mean(df[choice], conv_len=conv_len, future_frames = future_frames)

    if ignore_low_quality:
        select = df_smooth["low_quality"] == 1
        df_smooth.loc[select, choices] = 0
        df_smooth.loc[select, "low_quality"] = 1

    if db:
        for i, row in tqdm(df_smooth.iterrows()):
            _pred = [row[choice] for choice in choices]
            _labels = [choices[i] for i, v in enumerate(_pred) if v == 1]
            db.multilabel_prediction.update_one(
                {"image_id": row["image_id"]},
                {"$set": {"labels_smooth": _labels, "prediction_smooth": _pred}}
            )

    return df_smooth

def predictions_to_array(predictions, pred_smooth=False):
    image_id = []
    choices = None
    preds = []
    n_frame = []
    for p in predictions:
        image_id.append(p.image_id)
        if choices:
            assert choices == p.choices
        else:
            choices = p.choices
            
        n_frame.append(p.n)
        if pred_smooth:
            preds.append(p.prediction_smooth)
        else: 
            preds.append(p.prediction)

    preds = np.array(preds)
    preds[preds > 0.5] = 1
    preds[preds <= 0.5] = 0
    df = pd.DataFrame({"image_id": image_id, "n_frame": n_frame})
    for i, choice in enumerate(choices):
        df[choice] = preds[:, i]

    df = df.sort_values(by="n_frame")
    return df, choices

def map_df(df, wt_map_lookup):
    mapped_df = pd.DataFrame()
    mapped_df["image_id"] = df["image_id"]
    mapped_df["n_frame"] = df["n_frame"]

    for key, value in wt_map_lookup.items():
        if not value in mapped_df.columns:
            mapped_df[value] = 0
        if key in df.columns:
            select = df[key] == 1
            mapped_df.loc[select, value] = 1

    return mapped_df


def get_intervention_segments(df, labels, min_length=None):
    segments = {}
    for label in labels:
        ranges = []
        if label in df.columns:
            frame_list = df.loc[df.loc[:, label] == 1, "n_frame"].to_list()
            ranges = get_consecutive_ranges(frame_list)
            if min_length:
                ranges = [r for r in ranges if r[1]-r[0] >= min_length]
    
        segments[label] = ranges

    return segments

def merge_nearby_segments(range_tuples, max_diff):
    merged_ranges = []
    for r in range_tuples:
        if not merged_ranges:
            merged_ranges.append(r)
            continue

        if r[0] - merged_ranges[-1][1] <= max_diff:
            merged_ranges[-1] = (merged_ranges[-1][0], r[1])
        else:
            merged_ranges.append(r)

    return merged_ranges


def range_tuple_to_time(range_tuple, fps):
    return (range_tuple[1] - range_tuple[0]) / fps

def range_tuples_to_time(range_tuples, fps):
    return (range_tuples[-1][1] - range_tuples[0][0]) / fps