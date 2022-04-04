from ukw_tools.classes.report import ReportPolypAnnotationResult, ReportAnnotationResult
import streamlit as st

class StreamlitReportAnnotation:
    def __init__(self, report: ReportAnnotationResult):
            
        self.report = report
        self.base_expander = st.expander("Base", True)
        with self.base_expander:
            self.base_cols = st.columns(6)
        self.report = self.report_inputs()
        
    def polyp_report_inputs(self, i):
        _cols = st.columns(6)
        if i < len(self.report.polyps):
            _polyp_report = self.report.polyps[i]
        else:
            _polyp_report = ReportPolypAnnotationResult()

        p_report_annotation = {
            "location_segment": _cols[0].selectbox(
                "Location segment",
                _polyp_report.location_segment_options,
                index=_polyp_report.location_segment_options.index(
                    _polyp_report.location_segment
                ),
                key = f"polyp_{i}_location_segment"
            ),
            "location_cm": _cols[0].number_input(
                "Location (cm)", value=_polyp_report.location_cm,
                key = f"polyp_{i}_location_cm"
            ),
            "size_category": _cols[0].selectbox(
                "Size category",
                _polyp_report.size_category_options,
                index=_polyp_report.size_category_options.index(
                    _polyp_report.size_category
                ),
                key = f"polyp_{i}_size_category"
            ),
            "size_mm": _cols[0].number_input("Size (mm)", value=_polyp_report.size_mm, key = f"polyp_{i}_size_mm"),
            "surface_intact": _cols[1].selectbox(
                "Surface intact",
                _polyp_report.surface_intact_options,
                index=_polyp_report.surface_intact_options.index(
                    str(_polyp_report.surface_intact).lower()
                ),
                key = f"polyp_{i}_surface_intact"
            ),
            "paris": _cols[1].multiselect(
                "Paris", _polyp_report.paris_options, default=_polyp_report.paris,
                key = f"polyp_{i}_paris"
            ),
            "morphology": _cols[1].selectbox(
                "Morphology",
                _polyp_report.morphology_options,
                index=_polyp_report.morphology_options.index(_polyp_report.morphology),
                key = f"polyp_{i}_morphology"
            ),
            "nice": _cols[1].selectbox(
                "Nice",
                _polyp_report.nice_options,
                index=_polyp_report.nice_options.index(_polyp_report.nice),
                key = f"polyp_{i}_nice"
            ),
            "rating": _cols[2].selectbox(
                "Rating",
                _polyp_report.rating_options,
                index=_polyp_report.rating_options.index(_polyp_report.rating),
                key = f"polyp_{i}_rating"
            ),
            "lst": _cols[2].selectbox(
                "LST",
                _polyp_report.lst_options,
                index=_polyp_report.lst_options.index(_polyp_report.lst),
                key = f"polyp_{i}_lst"
            ),
            "injection": _cols[2].selectbox(
                "Injection",
                _polyp_report.injection_options,
                index=_polyp_report.injection_options.index(str(_polyp_report.injection).lower()),
                key = f"polyp_{i}_injection"
            ),
            "resection": _cols[3].checkbox("Resection", value=_polyp_report.resection, key = f"polyp_{i}_resection"),
        
            }
        if p_report_annotation["injection"] == "true":
            p_report_annotation.update({
                "non_lifting_sign": _cols[2].selectbox(
                    "Non lifting sign",
                    _polyp_report.non_lifting_sign_options,
                    index=_polyp_report.non_lifting_sign_options.index(
                        str(_polyp_report.non_lifting_sign).lower()
                    ),
                    key = f"polyp_{i}_non_lifting_sign"
                )
            })
        if p_report_annotation["resection"]:
            p_report_annotation.update({
                "tool": _cols[3].selectbox(
                    "Tool",
                    _polyp_report.tool_options,
                    index=_polyp_report.tool_options.index(_polyp_report.tool),
                    key = f"polyp_{i}_tool"
                ),
                "resection_technique": _cols[3].selectbox(
                    "Resection technique",
                    _polyp_report.resection_technique_options,
                    index=_polyp_report.resection_technique_options.index(
                        _polyp_report.resection_technique
                    ),
                    key = f"polyp_{i}_resection_technique"
                ),
                "ectomy_wound_care": _cols[4].selectbox(
                    "Ectomy wound care",
                    _polyp_report.ectomy_wound_care_options,
                    index=_polyp_report.ectomy_wound_care_options.index(
                        str(_polyp_report.ectomy_wound_care).lower()
                    ),
                    key = f"polyp_{i}_ectomy_wound_care"
                ),
                "salvage": _cols[5].selectbox(
                    "Salvage",
                    _polyp_report.salvage_options,
                    index=_polyp_report.salvage_options.index(str(_polyp_report.salvage).lower()),
                    key = f"polyp_{i}_salvage"
                ),
            })
            if p_report_annotation["ectomy_wound_care"] == "true":
                p_report_annotation.update({
                    "ectomy_wound_care_technique": _cols[4].selectbox("Ectomy wound care technique", _polyp_report.ectomy_wound_care_technique_options, index=_polyp_report.ectomy_wound_care_technique_options.index(_polyp_report.ectomy_wound_care_technique), key = f"polyp_{i}_ectomy_wound_care_technique")
                })
                if p_report_annotation["ectomy_wound_care_technique"] == "apc":
                    p_report_annotation.update({
                    "apc_watts": _cols[4].number_input(
                        "APC (Watts)", value=_polyp_report.apc_watts,
                        key = f"polyp_{i}_apc_watts"
                    )})
                elif p_report_annotation["ectomy_wound_care_technique"] == "clip":
                    p_report_annotation.update({
                        "number_clips": _cols[4].number_input(
                            "Number of clips", value=_polyp_report.number_clips,
                            key = f"polyp_{i}_number_clips"
                        )
                    })
            if p_report_annotation["salvage"] == "true":
                p_report_annotation.update({
                    "resection_status_microscopic": _cols[5].selectbox(
                        "Resection status (microscopic)",
                        _polyp_report.resection_status_microscopic_options,
                        index=_polyp_report.resection_status_microscopic_options.index(
                            _polyp_report.resection_status_microscopic
                        ),
                        key = f"polyp_{i}_resection_status_microscopic"
                    ),
                    "dysplasia": _cols[5].selectbox(
                        "Dysplasia",
                        _polyp_report.dysplasia_options,
                        index=_polyp_report.dysplasia_options.index(_polyp_report.dysplasia),
                        key = f"polyp_{i}_dysplasia"
                    ),
                    "histo": _cols[5].selectbox(
                        "Histo",
                        _polyp_report.histo_options,
                        index=_polyp_report.histo_options.index(_polyp_report.histo),
                        key = f"polyp_{i}_histo"
                    ),
                })
        else: 
            p_report_annotation.update({
                "no_resection_reason": _cols[3].selectbox(
                    "No resection reason",
                    _polyp_report.no_resection_reason_options,
                    index=_polyp_report.no_resection_reason_options.index(
                        _polyp_report.no_resection_reason
                    ),
                    key = f"polyp_{i}_no_resection_reason"
                ),
            })

        _polyp_report = ReportPolypAnnotationResult(**p_report_annotation)
        return _polyp_report

    def polyp_inputs(self):
        self.polyp_report_expanders = []
        polyp_reports = []
        for i in range(self.report.n_polyps):
            self.polyp_report_expanders.append(
                st.expander("Polyp {}".format(i + 1), True)
            )
            with self.polyp_report_expanders[i]:
                polyp_reports.append(self.polyp_report_inputs(i))

        return polyp_reports

    def report_inputs(self):
        bc = self.base_cols
        self.report_annotation = {
            "intervention_time": bc[0].number_input(
                "Intervention time (min)", value=self.report.intervention_time
            ),
            "withdrawal_time": bc[1].number_input(
                "Withdrawal time (min)", value=self.report.withdrawal_time
            ),
            "sedation": bc[2].selectbox("Sedation", self.report.sedation_options),
            "bbps_worst": bc[3].number_input(
                "BBPS (worst)", value=self.report.bbps_worst
            ),
            "bbps_total": bc[4].number_input(
                "BBPS (total)", value=self.report.bbps_total
            ),
            "indication": bc[5].selectbox("Indication", self.report.indication_options),
            "n_polyps": bc[0].number_input("N polyps", value=self.report.n_polyps),
            "n_adenoma": bc[1].number_input("N adenoma", value=self.report.n_adenoma),
            "other_pathologies": bc[2].checkbox(
                "Other pathologies", value=self.report.other_pathologies
            ),
            "mark_other": bc[2].checkbox("Mark other", value=self.report.mark_other),
            "polyps": self.report.polyps,
        }
        report = ReportAnnotationResult(**self.report_annotation)
        report.polyps = self.polyp_inputs()
        return report