import numpy as np
import pandas as pd
import pytest

from src.traffic_risk.weak_label import WEAK_LABEL_FEATURES, WeakLabelError, generate_weak_label


def _frame(rows=6):
    data = {column: np.linspace(0, 1, rows) for column in WEAK_LABEL_FEATURES}
    return pd.DataFrame(data)


def test_weak_label_is_deterministic_and_emits_all_rows():
    frame = _frame()
    first = generate_weak_label(frame)
    second = generate_weak_label(frame)
    pd.testing.assert_series_equal(first["risk_weak"], second["risk_weak"])
    assert len(first) == len(frame)
    assert {"cluster", "risk_weak"}.issubset(first.columns)


@pytest.mark.parametrize("bad", [np.nan, np.inf, "not-a-number"])
def test_weak_label_rejects_nonfinite_or_non_numeric(bad):
    frame = _frame()
    frame.loc[0, "n_ts"] = bad
    with pytest.raises(WeakLabelError):
        generate_weak_label(frame)


def test_weak_label_rejects_missing_and_small_input():
    frame = _frame(2)
    with pytest.raises(WeakLabelError, match="at least"):
        generate_weak_label(frame)
    frame = _frame()
    frame = frame.drop(columns=["max_box_area_person"])
    with pytest.raises(WeakLabelError, match="missing"):
        generate_weak_label(frame)

