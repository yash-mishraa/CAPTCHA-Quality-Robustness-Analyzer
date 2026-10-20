"""Deterministic grouped split assignment."""

from __future__ import annotations

import pytest

from captcha_quality.config import ConfigError, SplitSettings
from captcha_quality.splits import SplitAssignment, SplitError, assign_grouped_splits


def _settings(
    *,
    train: float = 0.6,
    val: float = 0.2,
    test: float = 0.2,
    seed: int = 5,
    min_sources: int = 1,
    stratify: bool = True,
) -> SplitSettings:
    return SplitSettings(
        ratios=(("train", train), ("val", val), ("test", test)),
        seed=seed,
        min_sources_per_split=min_sources,
        stratify_by_font=stratify,
    )


def _groups(per_font: int, fonts: int = 3) -> dict[str, str]:
    return {
        f"src-{font:02d}-{index:02d}": f"font_{font}"
        for font in range(fonts)
        for index in range(per_font)
    }


def test_every_group_assigned_exactly_once() -> None:
    groups = _groups(5)
    assignment = assign_grouped_splits(groups, _settings())
    assert set(assignment.assignments) == set(groups)
    assert assignment.split_of("src-00-00") in ("train", "val", "test")


def test_assignment_is_deterministic() -> None:
    groups = _groups(6)
    first = assign_grouped_splits(groups, _settings())
    second = assign_grouped_splits(groups, _settings())
    assert first.assignments == second.assignments
    assert first.counts == second.counts


def test_different_seed_changes_assignment() -> None:
    groups = _groups(6)
    first = assign_grouped_splits(groups, _settings(seed=1))
    second = assign_grouped_splits(groups, _settings(seed=2))
    assert first.assignments != second.assignments


def test_font_stratification_places_every_font_in_every_split() -> None:
    groups = _groups(10)
    assignment = assign_grouped_splits(groups, _settings())
    for split in ("train", "val", "test"):
        assert set(assignment.font_counts[split]) == set(groups.values())
        assert all(count >= 1 for count in assignment.font_counts[split].values())


def test_counts_follow_ratios_for_large_fonts() -> None:
    groups = _groups(20)
    assignment = assign_grouped_splits(groups, _settings())
    assert assignment.counts == {"train": 36, "val": 12, "test": 12}


def test_three_sources_are_repaired_to_one_per_split() -> None:
    groups = {"a": "font_0", "b": "font_0", "c": "font_0"}
    assignment = assign_grouped_splits(groups, _settings())
    assert assignment.counts == {"train": 1, "val": 1, "test": 1}


def test_minimum_split_size_is_enforced() -> None:
    groups = _groups(1)
    with pytest.raises(SplitError, match="min_sources_per_split"):
        assign_grouped_splits(groups, _settings(min_sources=2, stratify=False))


def test_too_few_groups_is_rejected() -> None:
    with pytest.raises(SplitError, match="no groups"):
        assign_grouped_splits({}, _settings())
    with pytest.raises(SplitError, match="3 splits"):
        assign_grouped_splits({"a": "font_0", "b": "font_0"}, _settings())


def test_infeasible_font_stratification_is_rejected() -> None:
    groups = _groups(2)
    with pytest.raises(SplitError, match="font-stratified splits need"):
        assign_grouped_splits(groups, _settings())


def test_unstratified_split_ignores_fonts() -> None:
    groups = _groups(5)
    assignment = assign_grouped_splits(groups, _settings(stratify=False))
    assert set(assignment.assignments) == set(groups)
    assert sum(assignment.counts.values()) == len(groups)


def test_missing_group_lookup_raises() -> None:
    assignment: SplitAssignment = assign_grouped_splits(_groups(3), _settings())
    with pytest.raises(SplitError, match="no split assignment"):
        assignment.split_of("src-does-not-exist")


def test_unknown_split_name_raises() -> None:
    with pytest.raises(ConfigError, match="unknown split"):
        _settings().ratio("holdout")
