from tools.run_s5_cpu_experiment import CONFIGS, _nms_traffic_lights, map_crop_box, select_development_samples


def test_development_selection_is_group_safe_and_keeps_requested_strata():
    rows = []
    for index in range(10):
        is_light = index < 7
        rows.append({"sample_id": f"sample-{index}", "group_id": f"ride-{index}",
                     "source_split": "val", "split": "development",
                     "objects": ([{"category": "traffic light", "tiny": index < 4,
                                   "bbox_size": "small" if index < 4 else "medium"}] if is_light else [])})
    selected = select_development_samples(rows, max_images=6, small_tl_target=4, nonlight_target=2)
    assert len(selected) == 6
    assert len({row["group_id"] for row in selected}) == 6
    assert sum(any(o["category"] == "traffic light" and (o["tiny"] or o["bbox_size"] == "small")
                   for o in row["objects"]) for row in selected) == 4
    assert sum(not any(o["category"] == "traffic light" for o in row["objects"])
               for row in selected) == 2
    assert selected == select_development_samples(rows, max_images=6, small_tl_target=4, nonlight_target=2)


def test_roi_merge_suppresses_overlapping_lights_but_keeps_nonlights():
    rows = [
        {"category": "traffic light", "bbox": [0, 0, 10, 10], "confidence": .8},
        {"category": "traffic light", "bbox": [1, 1, 11, 11], "confidence": .7},
        {"category": "car", "bbox": [1, 1, 11, 11], "confidence": .9},
    ]
    merged = _nms_traffic_lights(rows, .45)
    assert len(merged) == 2
    assert sum(row["category"] == "traffic light" for row in merged) == 1
    assert sum(row["category"] == "car" for row in merged) == 1


def test_plan_configurations_include_all_roi_sizes_and_thresholds():
    assert [(row["name"], row["imgsz"], row["confidence"], row["roi"]) for row in CONFIGS] == [
        ("B0", 640, .25, False), ("C1", 960, .25, False),
        ("C2", 640, .15, False), ("C3", 960, .15, False),
        ("C4-640", 640, .25, True), ("C4-960", 960, .25, True),
        ("C5-640", 640, .15, True), ("C5-960", 960, .15, True),
    ]


def test_roi_box_maps_to_original_coordinates_and_clips():
    assert map_crop_box((10, 20, 40, 60), 0, 0, 100, 100) == [10, 20, 40, 60]
    assert map_crop_box((-3, 5, 110, 120), 5, 10, 100, 100) == [2, 15, 100, 100]
