from __future__ import annotations


import pytest

from traffic_risk.build_features import compute_features


def obj(category, box, color="unknown"):
    return {"category": category, "box2d": dict(zip(("x1", "y1", "x2", "y2"), box, strict=True)), "attributes": {"trafficLightColor": color}}


def test_empty_scene_has_finite_zero_features():
    with pytest.raises(ValueError, match="no valid objects"):
        compute_features([], 100, 200)


def test_invalid_and_out_of_bounds_boxes_are_handled():
    result = compute_features([obj("person", (-10, 50, 50, 120)), obj("car", (9, 9, 1, 1)), obj("car", (0, 0, float("nan"), 2))], 100, 100)
    assert result["n_person"] == 1
    assert result["n_vehicle"] == 0
    assert result["bottom_half_person_ratio"] == 1


def test_counts_proximity_density_and_overlap():
    result = compute_features([obj("person", (20, 50, 80, 100)), obj("car", (10, 10, 90, 90)), obj("bicycle", (0, 0, 10, 10))], 100, 100)
    assert result["n_human"] == 2
    assert result["near_person_count"] == 1
    assert result["near_vehicle_count"] == 1
    assert result["mean_overlap_iou"] > 0


def test_unambiguous_center_light_is_selected():
    result = compute_features([obj("traffic light", (45, 15, 55, 35), "red")], 100, 100)
    assert result["central_tl_color"] == "red"
    assert result["central_tl_is_red"] == 1
    assert result["n_tl_red"] == 1


def test_edge_light_and_ambiguous_center_stay_unknown():
    edge = compute_features([obj("traffic light", (0, 0, 2, 2), "red")], 100, 100)
    assert edge["central_tl_color"] == "unknown"
    assert edge["central_tl_is_red"] == 0
    assert edge["n_tl_red"] == 1
    result = compute_features([obj("traffic light", (45, 10, 55, 25), "red"), obj("traffic light", (45, 11, 55, 26), "green")], 100, 100)
    assert result["central_tl_color"] == "unknown"


@pytest.mark.parametrize("height,width", [(0, 1), (1, 0), (-1, 2)])
def test_invalid_dimensions(height, width):
    with pytest.raises(ValueError):
        compute_features([], height, width)


def test_invalid_thresholds():
    with pytest.raises(ValueError):
        compute_features([], 10, 10, center_width_pct=2)
