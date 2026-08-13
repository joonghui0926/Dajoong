from __future__ import annotations

import numpy as np
import pytest

from buili_plan2bim.global_topology_training import (
    TopologyTrainOptions,
    _balanced_class_weights,
    _binary_macro_f1,
    _non_background_macro_f1,
    _split_records,
)


def test_topology_train_options_validate_release_safe_bounds() -> None:
    TopologyTrainOptions().validate()

    with pytest.raises(ValueError, match="epochs"):
        TopologyTrainOptions(epochs=0).validate()
    with pytest.raises(ValueError, match="validation_fraction"):
        TopologyTrainOptions(validation_fraction=0.0).validate()


def test_small_training_corpus_always_has_a_stable_validation_split() -> None:
    records = [{"sample_id": f"sample-{index}"} for index in range(3)]

    training, validation = _split_records(records, 0.1)

    assert len(training) == 2
    assert len(validation) == 1
    assert {item["sample_id"] for item in training}.isdisjoint(
        {item["sample_id"] for item in validation}
    )


def test_balanced_class_weights_upweight_rare_supported_classes() -> None:
    weights = _balanced_class_weights(
        [1_000_000, 100_000, 10_000, 0],
        ("background", "common", "rare", "absent"),
        background_weight=0.03,
    )

    assert weights[0] == pytest.approx(0.03)
    assert weights[2] > weights[1]
    assert weights[3] == 0.0


def test_validation_quality_metrics_penalize_missed_supported_classes() -> None:
    topology = _binary_macro_f1(
        np.asarray([10, 0]),
        np.asarray([0, 0]),
        np.asarray([0, 10]),
    )
    confusion = np.asarray(
        [
            [30, 0, 0],
            [0, 10, 0],
            [0, 10, 0],
        ]
    )

    assert topology == pytest.approx(0.5)
    assert _non_background_macro_f1(confusion) == pytest.approx(1 / 3)
