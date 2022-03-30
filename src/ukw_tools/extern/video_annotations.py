from bson import ObjectId
from ..classes.video_segmentation import VideoSegmentation

def convert_extern_annotation(annotation):
    # video_segmentation = {}
    annotation_by_label = {}
    label_annotation_index = {}
    dates = [_.date for _ in annotation]
    max_date = max(dates)

    # create a dict with all labels and the corresponding annotation index
    for i, _annotation in enumerate(annotation):
        for value in _annotation.flanks:
            if not value.name in label_annotation_index:
                label_annotation_index[value.name] = []
            label_annotation_index[value.name].append(i)

    # select the most recent annotation for each label
    for key in label_annotation_index.keys():
        label_annotation_index[key] = list(set(label_annotation_index[key]))
        indices = label_annotation_index[key]
        selected_dates = [dates[i] for i in indices]
        max_date_index = selected_dates.index(max(selected_dates))
        selected_index = indices[max_date_index]
        selected_annotation = annotation[selected_index]
        annotation_by_label[key] = selected_annotation

    # filter all flanks, so that only flanks of the corresponding label are in the dict entry
    flanks = []
    for key in annotation_by_label.keys():
        _annotation = annotation_by_label[key]
        annotator_id = _annotation.extern_annotator_id
        date = _annotation.date
        values = _annotation.flanks
        values = [_ for _ in values if _.name == key]
        values = [_.to_intern(
            source = "video_web_annotation",
            annotator_id = annotator_id,
            date = date
        ) for _ in values]
        flanks.extend(values)

    flanks.sort(key=lambda x: x.start)
    return flanks, max_date

def extern_to_intern_video_annotation(examination_id: ObjectId, annotation):
    flanks, max_date = convert_extern_annotation(annotation)
    segmentation_dict = {
        "examination_id": examination_id,
        "annotation": flanks
    }
    segmentation = VideoSegmentation(**segmentation_dict)

    return segmentation