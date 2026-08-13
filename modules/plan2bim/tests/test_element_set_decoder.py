from __future__ import annotations

from buili_plan2bim.core.model.aec_decode import PixelLineProposal, PixelSymbolProposal
from buili_plan2bim.element_set_decoder import decode_element_set


def _symbol(identifier: str, box, *, label: str = "sink", confidence: float = 0.9):
    return PixelSymbolProposal(
        id=identifier,
        symbol_class=label,
        center_px=((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
        bbox_px=box,
        confidence=confidence,
        uncertainty=1.0 - confidence,
        source_ref_ids=["source"],
        model_version="test",
        review_required=False,
    )


def _deferred_symbol(
    identifier: str,
    box,
    *,
    label: str = "base_cabinet",
    confidence: float = 0.6,
):
    return _symbol(
        identifier,
        box,
        label=label,
        confidence=confidence,
    ).model_copy(update={"model_version": "test+set-deferred-v1"})


def test_set_decoder_keeps_adjacent_cabinet_cells_but_suppresses_same_instance():
    first = _symbol("first", (10.0, 10.0, 30.0, 30.0), label="base_cabinet")
    duplicate = _symbol(
        "duplicate",
        (11.0, 11.0, 31.0, 31.0),
        label="base_cabinet",
        confidence=0.8,
    )
    adjacent = _symbol("adjacent", (31.0, 10.0, 51.0, 30.0), label="base_cabinet")

    selected, decisions = decode_element_set(
        [first, duplicate, adjacent],
        host_walls=[],
    )

    assert {item.id for item in selected} == {"first", "adjacent"}
    assert next(item for item in decisions if item.candidate_id == "duplicate").decision == (
        "same_instance_suppressed"
    )


def test_set_decoder_requires_wall_host_for_an_opening():
    window = _symbol("window", (20.0, 40.0, 40.0, 46.0), label="window")
    wall = PixelLineProposal(
        id="wall",
        start_px=(0.0, 43.0),
        end_px=(60.0, 43.0),
        thickness_px=4.0,
        confidence=0.9,
        uncertainty=0.1,
        source_ref_ids=["source"],
        model_version="test",
        review_required=False,
    )

    selected, _ = decode_element_set([window], host_walls=[wall])
    rejected, decisions = decode_element_set([window], host_walls=[])

    assert selected == [window]
    assert rejected == []
    assert decisions[0].decision == "wall_host_missing"


def test_set_decoder_resolves_conflicting_classes_for_the_same_geometry():
    sink = _symbol("sink", (10.0, 10.0, 40.0, 40.0), label="sink", confidence=0.92)
    jacuzzi = _symbol(
        "jacuzzi",
        (11.0, 11.0, 41.0, 41.0),
        label="jacuzzi",
        confidence=0.81,
    )

    selected, decisions = decode_element_set([jacuzzi, sink], host_walls=[])

    assert selected == [sink]
    assert next(item for item in decisions if item.candidate_id == "jacuzzi").decision == (
        "same_instance_suppressed"
    )


def test_set_decoder_keeps_a_tap_nested_inside_a_sink():
    sink = _symbol("sink", (10.0, 10.0, 50.0, 50.0), label="sink")
    tap = _symbol("tap", (24.0, 22.0, 34.0, 32.0), label="water_tap")

    selected, _ = decode_element_set([sink, tap], host_walls=[])

    assert {item.id for item in selected} == {"sink", "tap"}


def test_set_decoder_suppresses_full_size_tap_inside_an_appliance():
    appliance = _symbol(
        "appliance",
        (20.0, 20.0, 60.0, 70.0),
        label="electrical_appliance",
        confidence=0.95,
    )
    tap = _symbol(
        "tap",
        (12.0, 10.0, 68.0, 78.0),
        label="water_tap",
        confidence=0.70,
    )

    selected, _ = decode_element_set([tap, appliance], host_walls=[])

    assert selected == [appliance]


def test_set_decoder_merges_offset_hypotheses_for_one_linear_screen():
    first = _symbol(
        "first",
        (390.0, 920.0, 397.0, 1010.0),
        label="shower_screen",
        confidence=0.95,
    )
    second = _symbol(
        "second",
        (393.0, 920.0, 399.0, 1010.0),
        label="shower_screen",
        confidence=0.90,
    )

    selected, _ = decode_element_set([second, first], host_walls=[])

    assert selected == [first]


def test_set_decoder_merges_one_linear_instance_despite_class_disagreement():
    screen = _symbol(
        "screen",
        (390.0, 920.0, 397.0, 1010.0),
        label="shower_screen",
        confidence=0.95,
    )
    fragment = _symbol(
        "fragment",
        (393.0, 920.0, 399.0, 1010.0),
        label="misc",
        confidence=0.82,
    )

    selected, decisions = decode_element_set([fragment, screen], host_walls=[])

    assert selected == [screen]
    assert next(
        item for item in decisions if item.candidate_id == "fragment"
    ).decision == "same_instance_suppressed"


def test_set_decoder_merges_overlapping_same_family_plumbing_hypotheses():
    toilet = _symbol(
        "toilet",
        (400.0, 910.0, 472.0, 996.0),
        label="toilet",
        confidence=0.97,
    )
    false_tap = _symbol(
        "false-tap",
        (409.0, 919.0, 464.0, 949.0),
        label="water_tap",
        confidence=0.94,
    )

    selected, _ = decode_element_set([false_tap, toilet], host_walls=[])

    assert selected == [toilet]


def test_set_decoder_resolves_generic_housing_from_an_equipment_run():
    upper = _symbol(
        "upper",
        (10.0, 10.0, 40.0, 35.0),
        label="base_cabinet",
    )
    appliance = _symbol(
        "appliance",
        (10.0, 36.0, 40.0, 65.0),
        label="electrical_appliance",
    )
    ambiguous = _symbol(
        "ambiguous",
        (10.0, 67.0, 40.0, 96.0),
        label="housing",
    )

    selected, _ = decode_element_set(
        [upper, appliance, ambiguous],
        host_walls=[],
    )

    assert {item.id: item.symbol_class for item in selected}["ambiguous"] == (
        "base_cabinet"
    )


def test_set_decoder_uses_the_consensus_extent_not_only_peak_confidence():
    oversized = _symbol(
        "oversized",
        (8.0, 8.0, 48.0, 48.0),
        confidence=0.96,
    )
    consensus = _symbol(
        "consensus",
        (12.0, 12.0, 42.0, 42.0),
        confidence=0.93,
    )
    corroborating = _symbol(
        "corroborating",
        (13.0, 13.0, 43.0, 43.0),
        confidence=0.90,
    )

    selected, _ = decode_element_set(
        [oversized, corroborating, consensus],
        host_walls=[],
    )

    assert selected == [consensus]


def test_set_decoder_rescues_a_deferred_module_inside_a_supported_run():
    before = _symbol("before", (10.0, 10.0, 40.0, 40.0), label="base_cabinet")
    missing = _deferred_symbol("missing", (10.0, 42.0, 40.0, 72.0))
    after = _symbol("after", (10.0, 74.0, 40.0, 104.0), label="base_cabinet")

    selected, _ = decode_element_set([before, missing, after], host_walls=[])

    assert {item.id for item in selected} == {"before", "missing", "after"}


def test_set_decoder_rejects_an_isolated_deferred_candidate():
    anchor = _symbol("anchor", (10.0, 10.0, 40.0, 40.0), label="base_cabinet")
    isolated = _deferred_symbol("isolated", (100.0, 100.0, 130.0, 130.0))

    selected, decisions = decode_element_set([anchor, isolated], host_walls=[])

    assert selected == [anchor]
    assert next(
        item for item in decisions if item.candidate_id == "isolated"
    ).decision == "insufficient_set_support"
