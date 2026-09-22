"""Tests for ``features.registry``.

Each test is load-bearing: removing the guard it exercises must cause the test
to fail, not just produce a different number. This is documented on each test.

Tests cover:
* Mandatory warm-up: ``warmup_bars < 1`` raises at construction.
* Duplicate registration raises.
* Missing registration raises on ``get``.
* Definition hash is stable across identical definitions.
* Definition hash changes when any field changes.
* ``warmup_seconds`` is ``warmup_bars * interval_seconds``.
* ``max_warmup_seconds`` returns the largest warm-up across all registered defs.
* ``as_manifest_dict`` is sorted by name and JSON-serializable.
"""

from __future__ import annotations

import json
from typing import cast

import pytest

from memetrader.features.registry import FeatureDefinition, FeatureRegistry

# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


def test_warmup_bars_zero_raises() -> None:
    """warmup_bars < 1 must raise at construction.

    Guard: if ``__post_init__`` does not check ``warmup_bars``, this test would
    succeed with a warmup_bars=0 definition, which would then produce a
    warmup_seconds of 0 and silently allow a zero-length embargo.
    """
    with pytest.raises(ValueError, match="warmup_bars"):
        FeatureDefinition(
            name="bad_feature",
            version="1.0.0",
            inputs=("bars",),
            warmup_bars=0,
            prefix_safe=True,
        )


def test_empty_name_raises() -> None:
    with pytest.raises(ValueError, match="name"):
        FeatureDefinition(
            name="",
            version="1.0.0",
            inputs=("bars",),
            warmup_bars=1,
            prefix_safe=True,
        )


def test_empty_version_raises() -> None:
    with pytest.raises(ValueError, match="version"):
        FeatureDefinition(
            name="ok_feature",
            version="",
            inputs=("bars",),
            warmup_bars=1,
            prefix_safe=True,
        )


# ---------------------------------------------------------------------------
# Hash stability
# ---------------------------------------------------------------------------


def test_definition_hash_is_stable() -> None:
    """The same definition must always produce the same hash.

    Guard: if hashing uses id() or a non-deterministic source, the hash
    would differ between runs and the manifest diff would flag every run
    as non-equivalent even with no feature changes.
    """
    d1 = FeatureDefinition(
        name="my_feat",
        version="1.0.0",
        inputs=("bars", "snapshot"),
        warmup_bars=21,
        prefix_safe=True,
    )
    d2 = FeatureDefinition(
        name="my_feat",
        version="1.0.0",
        inputs=("bars", "snapshot"),
        warmup_bars=21,
        prefix_safe=True,
    )
    assert d1.definition_hash == d2.definition_hash
    assert len(d1.definition_hash) == 16  # truncated to 16 hex chars


def test_definition_hash_changes_on_version_bump() -> None:
    """A version bump must change the hash.

    Guard: if the version is not included in the hash payload, a logic change
    with a version bump would not be detectable in the manifest.
    """
    d1 = FeatureDefinition(
        name="my_feat",
        version="1.0.0",
        inputs=("bars",),
        warmup_bars=21,
        prefix_safe=True,
    )
    d2 = FeatureDefinition(
        name="my_feat",
        version="1.1.0",
        inputs=("bars",),
        warmup_bars=21,
        prefix_safe=True,
    )
    assert d1.definition_hash != d2.definition_hash


def test_definition_hash_changes_on_input_change() -> None:
    d1 = FeatureDefinition(
        name="my_feat",
        version="1.0.0",
        inputs=("bars",),
        warmup_bars=21,
        prefix_safe=True,
    )
    d2 = FeatureDefinition(
        name="my_feat",
        version="1.0.0",
        inputs=("bars", "snapshot"),
        warmup_bars=21,
        prefix_safe=True,
    )
    assert d1.definition_hash != d2.definition_hash


def test_definition_hash_input_order_independent() -> None:
    """Input order must not affect the hash.

    Guard: if inputs are not sorted before hashing, two definitions with the
    same inputs in different order would produce different hashes and be treated
    as different features.
    """
    d1 = FeatureDefinition(
        name="my_feat",
        version="1.0.0",
        inputs=("bars", "snapshot"),
        warmup_bars=21,
        prefix_safe=True,
    )
    d2 = FeatureDefinition(
        name="my_feat",
        version="1.0.0",
        inputs=("snapshot", "bars"),
        warmup_bars=21,
        prefix_safe=True,
    )
    assert d1.definition_hash == d2.definition_hash


# ---------------------------------------------------------------------------
# warmup_seconds
# ---------------------------------------------------------------------------


def test_warmup_seconds() -> None:
    """warmup_seconds is warmup_bars * interval_seconds.

    Guard: if warmup_seconds uses a hardcoded interval, the embargo would be
    wrong for non-default timeframes.
    """
    d = FeatureDefinition(
        name="rvol",
        version="1.0.0",
        inputs=("bars",),
        warmup_bars=21,
        prefix_safe=True,
    )
    assert d.warmup_seconds(3600.0) == pytest.approx(21 * 3600.0)
    assert d.warmup_seconds(300.0) == pytest.approx(21 * 300.0)


# ---------------------------------------------------------------------------
# Registry — duplicate and missing
# ---------------------------------------------------------------------------


def test_duplicate_registration_raises() -> None:
    """Registering the same name twice must raise.

    Guard: if the registry allows duplicates, the second definition silently
    shadows the first, and the manifest records whichever was registered last —
    which may not be the one actually used for computation.
    """
    reg = FeatureRegistry()
    d = FeatureDefinition(
        name="feat", version="1.0.0", inputs=(), warmup_bars=1, prefix_safe=True
    )
    reg.register(d)
    with pytest.raises(ValueError, match="already registered"):
        reg.register(d)


def test_get_missing_raises() -> None:
    """``get`` on an unregistered feature must raise.

    Guard: if ``get`` returns None or a default, a feature computed before
    registration would silently bypass the embargo check.
    """
    reg = FeatureRegistry()
    with pytest.raises(KeyError, match="not registered"):
        reg.get("nonexistent_feature")


def test_register_returns_definition() -> None:
    reg = FeatureRegistry()
    d = FeatureDefinition(
        name="feat", version="1.0.0", inputs=(), warmup_bars=1, prefix_safe=True
    )
    returned = reg.register(d)
    assert returned is d


# ---------------------------------------------------------------------------
# max_warmup_seconds
# ---------------------------------------------------------------------------


def test_max_warmup_seconds_returns_maximum() -> None:
    """max_warmup_seconds must return the longest warm-up, not the shortest.

    Guard: if min is used instead of max, the embargo would be based on the
    shortest warm-up and let longer-warm-up features see into the embargo period.
    """
    reg = FeatureRegistry()
    reg.register(
        FeatureDefinition(
            name="short", version="1.0.0", inputs=(), warmup_bars=5, prefix_safe=True
        )
    )
    reg.register(
        FeatureDefinition(
            name="long", version="1.0.0", inputs=(), warmup_bars=21, prefix_safe=True
        )
    )
    assert reg.max_warmup_seconds(3600.0) == pytest.approx(21 * 3600.0)


def test_max_warmup_empty_registry() -> None:
    reg = FeatureRegistry()
    assert reg.max_warmup_seconds(3600.0) == 0.0


# ---------------------------------------------------------------------------
# as_manifest_dict
# ---------------------------------------------------------------------------


def test_as_manifest_dict_sorted_by_name() -> None:
    """as_manifest_dict must be sorted by name.

    Guard: if unsorted, a reordering of feature registration (which can happen
    with refactoring) would change the manifest diff without any semantic change.
    """
    reg = FeatureRegistry()
    for name in ("zebra", "alpha", "middle"):
        reg.register(
            FeatureDefinition(
                name=name,
                version="1.0.0",
                inputs=(),
                warmup_bars=1,
                prefix_safe=True,
            )
        )
    manifest = reg.as_manifest_dict()
    # as_manifest_dict is typed dict[str, object] since entries mix strs, ints,
    # and bools; "name" is known (by construction above) to always be a str.
    names = [cast(str, d["name"]) for d in manifest]
    assert names == sorted(names)


def test_as_manifest_dict_is_json_serializable() -> None:
    reg = FeatureRegistry()
    reg.register(
        FeatureDefinition(
            name="feat", version="1.0.0", inputs=("bars",), warmup_bars=21, prefix_safe=True
        )
    )
    serialized = json.dumps(reg.as_manifest_dict())  # must not raise
    assert "feat" in serialized
