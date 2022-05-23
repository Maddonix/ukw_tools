ORIGIN_CATEGORIES = {
    "Boeck": ['Boeck-2', 'Boeck1', 'Boeck2', 'BoeckRetro', 'boeck2'],
    "Simonis": ['Simonis', 'Simonis_1', 'Simonis_2', 'simonis', 'simonis_1'],
    "Heil": ['HeilEndoBox', 'Heil_1', 'Heil_2', 'heil', 'heil_1', 'heil_2'],
    "Passek": ['PassekEndoBox', 'PassekRecordingPC', 'passek'],
    "Archive": ["archive_würzburg", "archive_stuttgart"],
    "Heubach": ["HeubachEndoBox","HeubachEndoBox (Kopie)", "heubach", "heubachendobox"],
    "Katharinen": ["KatharinenEndoBox", "Stuttgart"],
    "Ludwig": ["Ludwig", "Ludwig1", "ludwig1"],
    "gig_retrospective": ["2nd_round", "4th_round", "5th_round", "6th_round", "gi_genius_retrospective", "gi_genius_retrospective_frames", "gi_genius_ulm"],
    "Würzburg": ["ukw", "ukw_cwd", "Koloskopie", "koloskopie"]
}

ORIGIN_LOOKUP = {}
for key, value in ORIGIN_CATEGORIES.items():
    for v in value:
        ORIGIN_LOOKUP[v] = key

