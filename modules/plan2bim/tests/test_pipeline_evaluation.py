from __future__ import annotations

import pytest

from buili_plan2bim.pipeline_evaluation import (
    _aggregate_entity_classes,
    _entity_scores,
    _entity_type,
    _room_type,
    assert_prediction_not_trained_on_source,
    assert_prediction_scale_correspondence,
    assert_prediction_source_correspondence,
    prediction_source_hashes,
)


def test_room_type_uses_compiled_occupancy_before_display_name() -> None:
    assert _room_type({"name": "Bedroom 99", "occupancy": "bathroom"}) == "bathroom"
    assert _room_type({"name": "Outdoor 3"}) == "outdoor"
    assert _room_type({"room_type": "living_room"}) == "living"


def _target_fixture() -> dict[str, object]:
    return {
        "id": "target",
        "fixture_type": "Sink",
        "center_m": [50.0, 50.0],
        "polygon": [[30.0, 40.0], [70.0, 40.0], [70.0, 60.0], [30.0, 60.0]],
    }


def test_runtime_symbol_class_is_a_fixture_evaluation_type() -> None:
    assert _entity_type({"symbol_class": "electrical_appliance"}, kind="fixture") == (
        "appliance"
    )


def test_runtime_pixel_bbox_participates_in_geometry_aware_evaluation() -> None:
    prediction = {
        "id": "prediction",
        "symbol_class": "sink",
        "center_px": [50.0, 50.0],
        "bbox_px": [30.0, 40.0, 70.0, 60.0],
    }

    strict = _entity_scores(
        [prediction],
        [_target_fixture()],
        prediction_scale=1.0,
        kind="fixture",
        maximum_distance_px=32.0,
        minimum_dimension_similarity=0.75,
    )

    assert strict["f1"] == 1.0


def test_geometry_aware_entity_f1_rejects_right_center_but_wrong_size() -> None:
    prediction = {
        "id": "prediction",
        "type": "residential-sink",
        "center_m": [0.5, 0.5],
        "size_m": [0.1, 0.1, 0.9],
    }

    legacy = _entity_scores(
        [prediction],
        [_target_fixture()],
        prediction_scale=100.0,
        kind="fixture",
        maximum_distance_px=32.0,
    )
    strict = _entity_scores(
        [prediction],
        [_target_fixture()],
        prediction_scale=100.0,
        kind="fixture",
        maximum_distance_px=32.0,
        minimum_dimension_similarity=0.75,
    )

    assert legacy["f1"] == 1.0
    assert strict["f1"] == 0.0


def test_geometry_aware_entity_f1_accepts_matching_footprint() -> None:
    prediction = {
        "id": "prediction",
        "type": "residential-sink",
        "center_m": [0.5, 0.5],
        "size_m": [0.4, 0.2, 0.9],
    }

    strict = _entity_scores(
        [prediction],
        [_target_fixture()],
        prediction_scale=100.0,
        kind="fixture",
        maximum_distance_px=32.0,
        minimum_dimension_similarity=0.75,
    )

    assert strict["f1"] == 1.0
    assert strict["matched_pairs"][0]["dimension_similarity"] == 1.0
    assert strict["per_class"]["sink"]["f1"] == 1.0


def test_per_class_metric_prevents_easy_objects_from_hiding_a_missed_class() -> None:
    target = [
        _target_fixture(),
        {
            **_target_fixture(),
            "id": "bed",
            "fixture_type": "bed",
            "center_m": [90, 90],
        },
    ]
    prediction = [
        {
            "id": "sink-prediction",
            "type": "residential-sink",
            "center_m": [0.5, 0.5],
            "size_m": [0.4, 0.2, 0.9],
        }
    ]

    score = _entity_scores(
        prediction,
        target,
        prediction_scale=100.0,
        kind="fixture",
        maximum_distance_px=32.0,
        minimum_dimension_similarity=0.75,
    )
    aggregate = _aggregate_entity_classes(
        [{"fixtures_geometry_aware": score}],
        "fixtures_geometry_aware",
    )

    assert aggregate["sink"]["f1"] == 1.0
    assert aggregate["bed"]["f1"] == 0.0


@pytest.mark.parametrize(
    ("target_name", "model_name", "canonical"),
    (
        ("CoatRack", "coat_rack", "coatrack"),
        ("WaterTap", "water_tap", "watertap"),
        ("FireplaceCorner", "fireplace_corner", "fireplacecorner"),
        (
            "PlaceForFireplaceCorner",
            "place_for_fireplace_corner",
            "placeforfireplacecorner",
        ),
        ("WoodStove", "wood_stove", "woodstove"),
    ),
)
def test_fixture_taxonomy_normalizes_source_and_model_spelling(
    target_name: str,
    model_name: str,
    canonical: str,
) -> None:
    assert _entity_type({"fixture_type": target_name}, kind="fixture") == canonical
    assert _entity_type({"fixture_type": model_name}, kind="fixture") == canonical


def test_compiled_residential_family_prefix_does_not_split_fixture_class() -> None:
    assert (
        _entity_type({"family_id": "residential-water-tap"}, kind="fixture")
        == "watertap"
    )
    assert (
        _entity_type({"family_id": "residential-dining-table"}, kind="fixture")
        == "diningtable"
    )


@pytest.mark.parametrize(
    ("target_name", "model_name", "canonical"),
    (
        ("ElectricalAppliance Refrigerator", "refrigerator", "refrigerator"),
        ("ElectricalAppliance IntegratedStove", "stove", "stove"),
        ("ElectricalAppliance Dishwasher", "dishwasher", "dishwasher"),
        ("ElectricalAppliance WashingMachine", "washing_machine", "washingmachine"),
        ("ElectricalAppliance TumbleDryer", "tumble_dryer", "tumbledryer"),
        ("DiningTable", "dining_table", "diningtable"),
        ("CoffeeTable", "coffee_table", "coffeetable"),
    ),
)
def test_product_object_taxonomy_preserves_specific_identity(
    target_name: str,
    model_name: str,
    canonical: str,
) -> None:
    assert _entity_type({"fixture_type": target_name}, kind="fixture") == canonical
    assert _entity_type({"fixture_type": model_name}, kind="fixture") == canonical


def test_prediction_source_identity_is_read_from_exported_graph_contract() -> None:
    expected = "a" * 64
    graph = {
        "sources": [{"source_ref_id": "display-name", "source_hash": expected}],
        "provenance": {"source_image_sha256": expected},
    }

    assert prediction_source_hashes(graph) == {expected}
    assert_prediction_source_correspondence(
        graph,
        expected_source_sha256=expected,
        sheet_id="duplicate-friendly-name",
    )


def test_prediction_source_identity_rejects_same_name_different_drawing() -> None:
    with pytest.raises(ValueError, match="prediction/source mismatch"):
        assert_prediction_source_correspondence(
            {"sources": [{"source_hash": "b" * 64}]},
            expected_source_sha256="a" * 64,
            sheet_id="cubi-020",
        )


def test_evaluation_rejects_model_training_source_leakage() -> None:
    source_hash = "a" * 64
    prediction = {
        "pipeline": {"model_training_source_exclusions": [source_hash]}
    }

    with pytest.raises(ValueError, match="evaluation leakage"):
        assert_prediction_not_trained_on_source(
            prediction,
            expected_source_sha256=source_hash,
            sheet_id="cubi-014",
        )


def test_evaluation_accepts_unseen_source_hash() -> None:
    assert_prediction_not_trained_on_source(
        {"pipeline": {"model_training_source_exclusions": ["b" * 64]}},
        expected_source_sha256="a" * 64,
        sheet_id="cubi-014",
    )


def test_evaluation_rejects_silent_metric_scale_change() -> None:
    prediction = {
        "pipeline": {
            "metric_scale": {
                "pixels_per_meter": 100.0,
                "source": "user_supplied",
                "contract": "source_pixels_to_metric_bim_v1",
            }
        }
    }

    with pytest.raises(ValueError, match="prediction/scale mismatch"):
        assert_prediction_scale_correspondence(
            prediction,
            expected_pixels_per_meter=90.0,
            sheet_id="cubi-020",
        )


def test_evaluation_accepts_exported_metric_scale_contract() -> None:
    assert_prediction_scale_correspondence(
        {
            "pipeline": {
                "metric_scale": {
                    "pixels_per_meter": 100.0,
                    "source": "drawing_dimension",
                    "contract": "source_pixels_to_metric_bim_v1",
                }
            }
        },
        expected_pixels_per_meter=100.0,
        sheet_id="cubi-014",
    )
