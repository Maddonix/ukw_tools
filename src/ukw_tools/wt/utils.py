from bson import ObjectId
from ..classes.report import Report
import pandas as pd
import numpy as np
from collections import Counter

def get_comparisons(x_values, compare_labels, axes):
    comparisons = []
    for x_value in x_values:
        for axis in axes:
            for _labels in compare_labels:
                # if not axis:
                #     if "withdrawal_corrected" in _labels or "withdrawal_caecum_corrected" in _labels:
                #         st.write("LELLEK")
                #         continue
                comparison = dict(
                    x_values = [x_value, x_value],
                    color_values = _labels,
                    subplot = axis,
                    paired = True
                )
                comparisons.append(comparison)
    return comparisons

def set_none_to_nan(df):
    df = df.replace(to_replace=[None, -1, "unknown"], value=np.nan)
    return df

def count_by_origin(df):
    centers = df.origin_category.unique()
    categories = ["prediction", "annotation", "reported"]
    cat_count = {
        cat: len(df.loc[df.category == cat]) for cat in categories
    }

    cat_count_by_center = {
        center: {
            has_intervention: 
            len(df.loc[(df.origin_category == center) & (df.category == "reported")&(df.has_intervention == has_intervention)])
            for has_intervention in [True, False]
        }
        for center in centers
    }

    r = {
        "n": len(df.examination_id.unique()),
        "cat_count": cat_count,
        "cat_center_count": cat_count_by_center,
    }
    return r

def polyp_report_to_df_record(p_report, examination_id):
    p_record = p_report.dict()
    p_record = {key:value for key, value in p_record.items() if not "option" in key}
    p_record["examination_id"] = examination_id
    return p_record

def report_to_df_records(report):
    examination_id = report.examination_id
    r_record = report.report_annotation.dict()
    r_record["examination_id"] = examination_id
    p_records = report.report_annotation.polyps
    del r_record["polyps"]
    p_records = [polyp_report_to_df_record(p,examination_id) for p in p_records]

    return r_record, p_records

def get_wt_eval_df(test_ids, db, refresh = False):
    records = []
    e_records = []
    p_records = []
    test_video_keys = []

    for exam_id in test_ids:
        examination = db.get_examination(exam_id)
        video_key = examination.video_key
        test_video_keys.append(video_key)

        report = db.report.find_one({"examination_id": exam_id})
        report = Report(**report)

        evaluator = db.get_examination_evaluator(exam_id, refresh = refresh)
        report_summary = evaluator.report

        if report_summary["annotated_times"]:
            w_intervention = (report_summary["annotated_times"]["withdrawal"] !=
                report_summary["annotated_times"]["withdrawal_corrected"])
            c_intervention = (report_summary["annotated_times"]["caecum"] !=
                report_summary["annotated_times"]["caecum_corrected"])
            if w_intervention or c_intervention:
                has_intervention = True
            else: has_intervention = False

        else: has_intervention = None

        record_base={
                "examination_id": exam_id,
                "video_key": video_key,
                "origin_category": examination.origin_category,
                "has_intervention": has_intervention
            }
        
        if report_summary:
            predicted = report_summary["predicted_times"]
            if not predicted: predicted = {}
            predicted.update({"category":"prediction"})
            predicted.update(record_base)
            records.append(predicted)

            annotation = report_summary["annotated_times"]
            if not annotation: annotation = {}
            annotation.update({"category":"annotation"})
            annotation.update(record_base)
            records.append(annotation)

        reported_time = {}
        reported_time["withdrawal"] = None
        if report.report_annotation:
            _ = float(report.report_annotation.withdrawal_time*60)
            if _ > 0:
                reported_time["withdrawal"] = _
        if not reported_time["withdrawal"]:
            try:
                _segmentation = db.get_examination_segmentation_annotation(exam_id)
                if _segmentation:
                    wt0, wt1 = _segmentation.calculate_wt_from_freezes()
                    reported_time["withdrawal"] = wt0
                else:
                    reported_time["withdrawal"] = None
            except:
                print("INVALID")
                reported_time["withdrawal"] = None
            
        reported_time.update({"category":"reported"})
        reported_time.update(record_base)
        records.append(reported_time)

        if report.report_annotation:
            _e_record, _p_records = report_to_df_records(report)
            e_records.append(_e_record)
            p_records.extend(_p_records)
        else:
            print(f"No report annotation for {exam_id}")

    df = pd.DataFrame.from_records(records)
    df["withdrawal_caecum"] = df["withdrawal"] + df["caecum"]
    df["withdrawal_caecum_corrected"] = df["withdrawal_corrected"] + df["caecum_corrected"]
    e_report_df = pd.DataFrame.from_records(e_records)
    p_report_df = pd.DataFrame.from_records(p_records)
    # st.write(df)
    df["age"] = df.examination_id.apply(lambda x: db.get_examination(ObjectId(x)).age)
    df["gender"] = df.examination_id.apply(lambda x: db.get_examination(ObjectId(x)).gender)
    e_report_df["age"] = e_report_df.examination_id.apply(lambda x: db.get_examination(ObjectId(x)).age)
    e_report_df["gender"] = e_report_df.examination_id.apply(lambda x: db.get_examination(ObjectId(x)).gender)

    df = set_none_to_nan(df)
    e_report_df = set_none_to_nan(e_report_df)
    p_report_df = set_none_to_nan(p_report_df)
    
    return df, e_report_df, p_report_df

def report_df_summary(r_df, p_df):
    n_examination = len(r_df.examination_id.unique())
    n_exam_w_polyp = len(r_df.loc[r_df.n_polyps > 0])
    n_exam_w_adenoma = len(r_df.loc[r_df.n_adenoma > 0])
    _indication =r_df[~r_df.indication.isna()].indication.tolist()
    indication = Counter(_indication).most_common()
    bbps_mean = r_df.bbps_total.mean()
    mean_wt_reported = r_df.withdrawal_time.mean()

    n_polyp = len(p_df)
    histo = p_df.loc[~p_df.histo.isna()].histo.tolist()
    n_histo = len(histo)
    histo_count = Counter(histo).most_common()
    instrument_count = Counter(p_df.loc[~p_df.tool.isna()].tool).most_common()
    polyp_size_count = Counter(p_df.loc[~p_df.size_category.isna()].size_category).most_common()
    _paris = p_df.loc[~p_df.paris.isna()].paris.tolist()
    _paris = [_ for sublist in _paris for _ in sublist]
    paris_count = Counter(_paris).most_common()
    location_count = Counter(p_df.loc[~p_df.location_segment.isna()].location_segment).most_common()

    r = {
        "n_examination": n_examination,
        "n_exam_w_polyp": n_exam_w_polyp,
        "n_exam_w_adenoma": n_exam_w_adenoma,
        "pdr": n_exam_w_polyp/n_examination,
        "adr": n_exam_w_adenoma/n_examination,
        "indication": indication,
        "bbps_mean": bbps_mean,
        "bbps_std":  np.std(r_df.bbps_total),
        "mean_wt_reported": mean_wt_reported,
        "str_wt_reported": np.std(r_df.withdrawal_time),
        "n_polyp": n_polyp,
        "n_polyp_resected": p_df.resection.sum(),
        "intrument_count": instrument_count,
        "n_histo": n_histo,
        "histo_count": histo_count,
        "polyp_size_count": polyp_size_count,
        "paris_count": paris_count,
        "location_count": location_count,
        "gender": {
            "m": r_df.gender.sum(),
            "f": len(r_df) - r_df.gender.sum()
        },
        "age_mean": r_df.age.mean(),
        "age_std": np.std(r_df.age)
    }

    return r