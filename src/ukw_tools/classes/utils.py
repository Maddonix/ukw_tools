POLYP_EVALUATION_STRUCTURE = {
        "location_segment": {
        "attribute": "location_segment",
        "required": True,
        "required_if": []
        },
        "size_category": {
        "attribute": "size_category",
        "required": True,
        "required_if": []
        },
        "surface_intact": {
        "attribute": "surface_intact",
        "required": True,
        "required_if": [],
        },
        "rating": {
        "attribute": "rating",
        "required": True,
        "required_if": [],
        },
        "resection": {
        "attribute": "resection",
        "required": True,
        "required_if": [],
        },
        "location_cm": {
        "attribute": "location_cm",
        "required": False,
        "required_if": [
            {
            "attribute": "location_segment",
            "values": [
                "rectum",
                "sigma"
            ]
            }
        ],
        },
        "size_mm": {
        "attribute": "polyp_size_mm",
        "required": False,
        "required_if": []
        },
        "paris": {
        "attribute": "polyp_paris",
        "required": False,
        "required_if": [
            {
            "attribute": "size_category",
            "values": [
                "<5",
                "5-10",
                ">10-20",
                ">20"
            ]
            }
        ]
        },
        "lst": {
        "attribute": "lst",
        "required": False,
        "required_if": [
            {
            "attribute": "size_category",
            "values": [
                ">10-20",
                ">20"
            ]
            }
        ],
        },
        "nice": {
        "attribute": "nice",
        "required": False,
        "required_if": [
            {
            "attribute": "size_category",
            "values": [
                "5-10",
                ">10-20",
                ">20"
            ]
            }
        ],
        },
        "non_lifting_sign": {
        "attribute": "non_lifting_sign",
        "required": False,
        "required_if": [],
        },
        "tool": {
        "attribute": "tool",
        "required": False,
        "required_if": [
            {
            "attribute": "resection",
            "values": [
                True
            ]
            }
        ],
        },
        # "resection_technique": {
        # "attribute": "resection_technique",
        # "required": False,
        # "required_if": [
        #     {
        #     "attribute": "resection",
        #     "values": [
        #         True
        #     ]
        #     }
        # ],
        # },
        "salvage": {
        "attribute": "salvage",
        "required": False,
        "required_if": [
            {
            "attribute": "resection",
            "values": [
                True
            ]
            }
        ],
        },
        "ectomy_wound_care": {
        "attribute": "ectomy_wound_care",
        "required": False,
        "required_if": [
            {
            "attribute": "resection",
            "values": [
                True
            ]
            }
        ],
        },
        "ectomy_wound_care_technique": {
        "attribute": "ectomy_wound_care_technique",
        "required": False,
        "required_if": [
            {
            "attribute": "ectomy_wound_care",
            "values": [
                True
            ]
            }
        ],
        },
        "apc_watts": {
        "attribute": "apc_watts",
        "required": False,
        "required_if": [
            {
            "attribute": "ectomy_wound_care_technique",
            "values": [
                "apc"
            ]
            }
        ],
        },
        "number_clips": {
        "attribute": "number_clips",
        "required": False,
        "required_if": [
            {
            "attribute": "ectomy_wound_care_technique",
            "values": [
                "clip"
            ]
            }
        ],
        },
        "ectomy_wound_care_success": {
        "attribute": "ectomy_wound_care_success",
        "required": False,
        "required_if": [
            {
            "attribute": "ectomy_wound_care",
            "values": [
                True
            ]
            }
        ],
        },
        "no_resection_reason": {
        "attribute": "no_resection_reason",
        "required": False,
        "required_if": [
            {
            "attribute": "resection",
            "values": [
                False
            ]
            }
        ],
        },
        "resection_technique": {
        "attribute": "resection_technique",
        "required": False,
        "required_if": [
            {
            "attribute": "resection",
            "values": [
                True
            ]
            }
        ]
        }
        }
        