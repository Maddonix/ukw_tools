from typing import Dict, List, Optional, Union

from pydantic import BaseModel, validator

from ..media.text import split_size_str
from .base import PyObjectId
from .utils import POLYP_EVALUATION_STRUCTURE


class ReportPolypAnnotationResult(BaseModel):
    location_segment_options = [
        "unknown",
        "rectum",
        "sigma",
        "descendens",
        "transversum",
        "ascendens",
        "caecum",
        "terminal_ileum",
        "neoterminal_ileum",
        "right_flexure",
        "left_flexure",
        "right_colon",
        "left_colon"
    ]
    location_segment: str = "unknown"
    location_cm: int = -1
    size_category_options = ["<5", "5-10", ">10-20", ">20", "unknown"]
    size_category: str = "unknown"
    size_mm: int = -1
    surface_intact_options = ["true", "false", "unknown"]
    surface_intact: Union[bool, str] = "unknown"
    rating_options =[
        "unknown",
        "hyperplastic",
        "adenoma",
        "ssa",
        "inflammatory",
        "dalm",
        "malign"
    ]
    rating: str = "unknown"
    paris_options = [
        "unknown",
        "Is",
        "Ip",
        "Ix",
        "IIa",
        "IIb",
        "IIc",
        "IIx"
    ]        
    paris: List[str] = []
    dysplasia_options = [
            "unknown", "no", "low", "high"
        ]
    dysplasia: str = "unknown"
    histo_options =   ["unknown", "non_adenoma", "tubular_adenoma", "tubulovillous_adenoma", "sessil_serrated_lesion", "carcinoma", "not_evaluated"]
    histo: str = "unknown"
    morphology_options = [
        "unknown",
        "sessil",
        "flach",
        "gestielt"
    ]
    morphology: str = "unknown"
    nice_options = ["ASDASD", "unknown"]
    nice: str = "unknown"
    lst_options = ["ASDASD", "unknown"]
    lst: str = "unknown"
    non_lifting_sign_options = ["true", "false", "unknown"]
    non_lifting_sign: Union[bool, str] = "unknown"
    injection_options = ["true", "false", "unknown"]
    injection: Union[bool, str] = "unknown"

    tool_options = ["unknown", "grasper", "sling_hot", "sling_cold", "sling"]
    tool: str = "unknown"
    resection_technique_options = ["enbloc", "piecemeal", "biopsy", "unknown"]
    resection_technique: str = "unknown"
    resection_status_microscopic_options = ["R0", "R1", "R2", "unknown"]
    resection_status_microscopic: str = "unknown"
    salvage_options = ["true", "false", "unknown"]
    salvage: Union[bool, str] = "unknown"

    apc_watts: int = -1
    number_clips: int = -1

    ectomy_wound_care_sucess_options = ["unknown", "preventive", "hemostasis", "no_hemostasis", "reactivation_hemostasis", "reactivation_no_hemostasis"]
    ectomy_wound_care_success: str = "unknown"
    ectomy_wound_care_technique_options = ["unknown", "clip", "apc"]
    ectomy_wound_care_technique: str = "unknown"
    ectomy_wound_care_options = ["true", "false", "unknown"]
    ectomy_wound_care: Union[bool, str] = "unknown"

    no_resection_reason_options = ["unknown", "provided"]
    no_resection_reason: str = "unknown"

    resection: bool = False

    polyp_id: Optional[PyObjectId]

    def __hash__(self):
        return hash(repr(self))

    # @validator("paris", allow_reuse=True)
    # def val_paris(cls, v):
    #     if v == []:
    #         v = None
    #     return v

    # @validator("size_mm", allow_reuse=True)
    # def val_size_mm(cls, v, values):
    #     if v:
    #         if v < 0:
    #             v = None
    #         else:
    #             _range = split_size_str(values["size_category"])
    #             assert _range
    #             assert _range[0] <= v <= _range[1]

    #     return v

    # @validator("number_clips", "apc_watts", allow_reuse=True)
    # def val_int(cls, v):
    #     if v:
    #         if v < 0:
    #             v = None
    #         return v

    # @validator("resection", allow_reuse=True)
    # def val_resection(cls, v, values):
    #     req_resection = [
    #         "resection_technique",
    #         "resection_status_microscopic",
    #         "salvage",
    #         "ectomy_wound_care",  # if true, ectomy_wound_care_success
    #     ]

    #     req_no_resection = ["no_resection_reason"]

    #     if v:
    #         for req in req_resection:
    #             assert req in values

    #     elif v is False:
    #         for req in req_no_resection:
    #             assert req in values

    #     return v

    # @validator("ectomy_wound_care", allow_reuse=True)
    # def val_ectomy_wound_care(cls, v, values):
    #     req_wound_care = ["ectomy_wound_care_success", "ectomy_wound_care_technique"]
    #     if v:
    #         for req in req_wound_care:
    #             assert req in values

    #     return v

    # @validator("ectomy_wound_care_technique", allow_reuse=True)
    # def val_ectomy_technique(cls, v, values):
    #     if v == "apc":
    #         assert "apc_watts" in values
    #     elif v == "clip":
    #         assert "number_clips" in values
    #     return v

    def evaluate(self):
        polyp_report = self.dict()
        result = {"required": [], "optional": [], "found": []}

        for attribute, element in POLYP_EVALUATION_STRUCTURE.items():
            if element["required"]:
                result["required"].append(attribute)

            else:
                required = False
                for requirement in element["required_if"]:
                    required_attribute = requirement["attribute"]
                    if polyp_report[required_attribute] in requirement["values"]:
                        required = True
                        break
                if required:
                    result["required"].append(attribute)
                else:
                    result["optional"].append(attribute)

        for key, value in polyp_report.items():
            if not value == None and not value == "unknown" and not value == -1:
                result["found"].append(key)

        result["required_found"] = [
            _ for _ in result["required"] if _ in result["found"]
        ]
        result["optional_found"] = [
            _ for _ in result["optional"] if _ in result["found"]
        ]

        result["required_missing"] = [
            _ for _ in result["required"] if _ not in result["found"]
        ]
        result["optional_missing"] = [
            _ for _ in result["optional"] if _ not in result["found"]
        ]

        return result


class ReportAnnotationResult(BaseModel):
    polyps: List[ReportPolypAnnotationResult] = []
    intervention_time: int = -1
    withdrawal_time: int = -1
    sedation_options = ["unknown", "no", "propofol", "midazolam", "propofol+midazolam", "other"]
    sedation: str = "unknown"
    bbps_worst: int = -1
    bbps_total: int = -1
    n_polyps: int = 0
    n_adenoma: int = 0
    other_pathologies: bool = False
    indication_options = [
        "screening",
        "surveillance",
        "symptomatic",
        "other",
        "unknown",
    ]
    indication: str = "unknown"
    mark_other: bool = False

    def __hash__(self):
        return hash(repr(self))

    # @validator("polyps", allow_reuse=True, pre=True)
    # def dict_to_list(cls, v):
    #     if isinstance(v, dict):
    #         v = [__ for _, __ in v.items()]
    #     return v

    # @validator(
    #     "intervention_time",
    #     "withdrawal_time",
    #     "bbps_worst",
    #     "bbps_total",
    #     "n_polyps",
    #     allow_reuse=True,
    # )
    # def del_unknown(cls, v):
    #     if v:
    #         if v < 0:
    #             v = None
    #     return v

    # @validator("bbps_worst", allow_reuse=True)
    # def bbps_worst_range(cls, v):
    #     if v:
    #         assert v <= 3
    #     return v

    # @validator("bbps_total", allow_reuse=True)
    # def bbps_total_range(cls, v):
    #     if v:
    #         assert v <= 9
    #     return v

    # @validator("n_adenoma", allow_reuse=True)
    # def set_n_adenoma(cls, v, values):
    #     return v


class Report(BaseModel):
    examination: Optional[str]
    histo: Optional[str]
    examination_structured: Optional[Dict[str, str]]
    histo_structured: Optional[Dict[str, str]]
    report_annotation: Optional[ReportAnnotationResult]
    examination_id: PyObjectId
    id_extern: Optional[int]
