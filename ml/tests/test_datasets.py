from __future__ import annotations

import pytest

from satquery_ml.datasets import (
    assert_image_disjoint_splits,
    balance_task_examples,
    convert_box,
)


def test_vrsbench_coordinate_conversion_round_trip():
    source = [12.5, 20.0, 81.2, 95.0]
    converted = convert_box(source, source_max=100, destination_max=1000)
    restored = [value / 10 for value in converted]
    assert restored == pytest.approx(source, abs=0.05)


def test_invalid_box_fails():
    with pytest.raises(ValueError, match="ordered"):
        convert_box([80, 10, 20, 50])


def test_image_split_leakage_fails():
    with pytest.raises(ValueError, match="leakage"):
        assert_image_disjoint_splits(
            {"train": [{"image": "a.png"}], "test": [{"image": "a.png"}]}
        )


def test_balancing_is_deterministic():
    pools = {
        "vqa": [{"id": f"v{index}"} for index in range(10)],
        "grounding": [{"id": f"g{index}"} for index in range(10)],
        "caption": [{"id": f"c{index}"} for index in range(10)],
    }
    mix = {"vqa": 0.5, "grounding": 0.3, "caption": 0.2}
    first = balance_task_examples(pools, mix, 10, seed=42)
    second = balance_task_examples(pools, mix, 10, seed=42)
    assert first == second
