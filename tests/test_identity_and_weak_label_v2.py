import pandas as pd
import pytest

from src.sample_identity import IdentityConflict, SampleRecord, assert_destinations_free, validate_records
from src.weak_label import WEAK_LABEL_FEATURES, generate_weak_label


def _frame(n=6):
    data = {name: [float(i + (2 if name == "n_ts" else 0)) for i in range(n)] for name in WEAK_LABEL_FEATURES}
    return pd.DataFrame(data)


def test_weak_label_is_deterministic_and_has_three_classes():
    first = generate_weak_label(_frame())
    second = generate_weak_label(_frame())
    assert first["risk_weak"].tolist() == second["risk_weak"].tolist()
    assert set(first["risk_weak"]) == {0, 1, 2}


@pytest.mark.parametrize("bad", [{"n_ts": None}, {"max_box_area_person": float("inf")}])
def test_weak_label_rejects_invalid_values(bad):
    frame = _frame()
    for key, value in bad.items():
        frame.loc[0, key] = value
    with pytest.raises(ValueError, match="non-finite"):
        generate_weak_label(frame)


def test_weak_label_rejects_missing_and_small_input():
    frame = _frame(2)
    with pytest.raises(ValueError, match="At least"):
        generate_weak_label(frame)
    with pytest.raises(ValueError, match="Missing"):
        generate_weak_label(frame.drop(columns=["n_ts"]))


def test_identity_detects_conflicting_labels_and_destination(tmp_path):
    records = [
        SampleRecord("a.png", "a.png", "deadbeef", label="red"),
        SampleRecord("b.png", "b.png", "deadbeef", label="green"),
    ]
    with pytest.raises(IdentityConflict, match="conflicting labels"):
        validate_records(records)
    target = tmp_path / "out.png"
    target.write_bytes(b"x")
    with pytest.raises(IdentityConflict, match="Destination"):
        assert_destinations_free([target])

