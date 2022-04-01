from pathlib import Path

def process_smartie_record(r):
    _r = dict(
        adenoma_count = r["adenomaCount"],
        polyp_count = r["polypCount"],
        age_at_examination=r["ageDuringExamination"],
        video_date = r["videoCreationDate"],
        birthdate=r["birthdate"],
        study_id = list(r["studies"].keys())[0],
        video_key = Path(r["videoPath"]).name
    )
    _id = _r["study_id"]
    _r.update(r["studies"][_id])
    for questionnaire_name, value in r["questionnaires"].items():
        _r.update(value["questionAnswers"])


    return _r