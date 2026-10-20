"""Deterministic, group-aware train/val/test split assignment.

Splits are assigned to *groups* (one group per clean source image), so every
variant of a source always lands in the same split. Assignment happens before
any pixel is generated: the strategy is part of the committed configuration, not
a post-hoc choice. When font stratification is enabled, each font is shuffled
and allocated independently so that every split sees every font whenever the
configured source count allows it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np

from captcha_quality.config import SPLIT_NAMES, SplitSettings
from captcha_quality.corruptions import derive_noise_seed


class SplitError(ValueError):
    """Raised when a split policy cannot be satisfied by the available groups."""


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """Result of assigning every group to exactly one split."""

    assignments: Mapping[str, str]
    counts: Mapping[str, int]
    font_counts: Mapping[str, Mapping[str, int]]

    def split_of(self, group_id: str) -> str:
        """Return the split assigned to ``group_id``."""
        try:
            return self.assignments[group_id]
        except KeyError as exc:
            raise SplitError(f"group {group_id!r} has no split assignment") from exc


def _allocate(n_groups: int, settings: SplitSettings) -> dict[str, int]:
    """Allocate ``n_groups`` across splits by largest remainder.

    Every split receives at least one group when ``n_groups`` is at least the
    number of splits. Without that repair, a 3-group font-stratified allocation
    under 0.6/0.2/0.2 would give ``2/1/0`` and silently exclude a split from a
    font. The repair takes the extra group from the largest split, which keeps
    the allocation as close to the configured ratios as the group count allows.
    """
    raw = {name: settings.ratio(name) * n_groups for name in SPLIT_NAMES}
    counts = {name: math.floor(raw[name]) for name in SPLIT_NAMES}
    remaining = n_groups - sum(counts.values())
    order: Final[tuple[str, ...]] = SPLIT_NAMES
    by_remainder = sorted(
        order,
        key=lambda name: (-(raw[name] - math.floor(raw[name])), order.index(name)),
    )
    for name in by_remainder[:remaining]:
        counts[name] += 1

    while n_groups >= len(SPLIT_NAMES):
        empty = [name for name in SPLIT_NAMES if counts[name] == 0]
        if not empty:
            break
        donor = max(SPLIT_NAMES, key=lambda name: (counts[name], -SPLIT_NAMES.index(name)))
        counts[donor] -= 1
        counts[empty[0]] = 1

    return counts


def _shuffle(ids: list[str], *, seed_parts: tuple[object, ...]) -> list[str]:
    rng = np.random.default_rng(derive_noise_seed(*seed_parts))
    indices = rng.permutation(len(ids))
    return [ids[int(index)] for index in indices]


def assign_grouped_splits(
    groups: Mapping[str, str],
    settings: SplitSettings,
) -> SplitAssignment:
    """Assign each group to exactly one split.

    Parameters
    ----------
    groups:
        Mapping ``group_id -> font_id``. ``group_id`` is normally the
        ``source_id``; the font is used only for stratified allocation.
    settings:
        Configured ratios, seed, minimum size and stratification flag.

    Raises
    ------
    SplitError
        If there are too few groups, a split would be empty, a split would fall
        below ``min_sources_per_split``, or per-font allocation cannot put every
        font in every split as requested.
    """
    if not groups:
        raise SplitError("cannot assign splits: no groups were provided")
    if len(groups) < len(SPLIT_NAMES):
        raise SplitError(f"cannot assign {len(SPLIT_NAMES)} splits from {len(groups)} group(s)")

    assignments: dict[str, str] = {}
    font_counts: dict[str, dict[str, int]] = {
        split: {font: 0 for font in sorted(set(groups.values()))} for split in SPLIT_NAMES
    }
    group_ids = sorted(groups)

    if settings.stratify_by_font:
        fonts = sorted(set(groups.values()))
        for font in fonts:
            members = [group_id for group_id in group_ids if groups[group_id] == font]
            if len(members) < len(SPLIT_NAMES):
                raise SplitError(
                    f"font {font!r} has {len(members)} source(s); font-stratified splits need "
                    f"at least {len(SPLIT_NAMES)} so every split receives that font"
                )
            counts = _allocate(len(members), settings)
            empty = [name for name in SPLIT_NAMES if counts[name] == 0]
            if empty:
                raise SplitError(
                    f"split ratios {[settings.ratio(name) for name in SPLIT_NAMES]} are too "
                    f"skewed to place font {font!r} in {empty} with {len(members)} sources"
                )
            shuffled = _shuffle(members, seed_parts=(settings.seed, "split", font))
            cursor = 0
            for split in SPLIT_NAMES:
                for group_id in shuffled[cursor : cursor + counts[split]]:
                    assignments[group_id] = split
                    font_counts[split][font] += 1
                cursor += counts[split]
    else:
        counts = _allocate(len(group_ids), settings)
        shuffled = _shuffle(group_ids, seed_parts=(settings.seed, "split", "global"))
        cursor = 0
        for split in SPLIT_NAMES:
            for group_id in shuffled[cursor : cursor + counts[split]]:
                assignments[group_id] = split
            cursor += counts[split]

    if set(assignments) != set(groups):
        missing = sorted(set(groups) - set(assignments))
        raise SplitError(f"internal split error: groups without assignment: {missing}")

    totals = {
        split: sum(1 for value in assignments.values() if value == split) for split in SPLIT_NAMES
    }
    for split in SPLIT_NAMES:
        if totals[split] < settings.min_sources_per_split:
            raise SplitError(
                f"split {split!r} received {totals[split]} group(s), below "
                f"min_sources_per_split={settings.min_sources_per_split}; add sources or adjust ratios"
            )

    return SplitAssignment(
        assignments=assignments,
        counts=totals,
        font_counts={split: dict(counts) for split, counts in font_counts.items()},
    )
