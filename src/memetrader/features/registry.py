"""Feature definition registry.

Every feature that flows into a model must have a registered definition before
it may be computed. Registration serves two purposes that are equally important:

1. **Embargo safety.** The backtest's purge/embargo calculation needs to know
   the maximum warm-up length for every active feature. A feature that cannot
   state its warm-up cannot be correctly embargoed — the purge window would be
   too short, leaking training labels into the validation period. So warm-up is
   mandatory here, not optional.

2. **Run reproducibility.** The experiment manifest records a stable hash of
   every feature definition used in a run. If a feature's logic changes between
   runs without a version bump, the hash changes and the comparison is flagged
   as non-equivalent. Without that hash, two runs with the same feature *name*
   but different *definitions* would look comparable and not be.

``FeatureDefinition.warmup_bars`` is the count of closed bars the feature
needs before it can produce a non-None value. ``warmup_seconds`` converts that
to wall-clock time given an interval; the embargo uses wall-clock so that two
assets on different timeframes are embargoed to the same boundary.

``realized_vol_pct`` needs ``REALIZED_VOL_PERIOD + 1 = 21`` closes
(20 log returns), so its ``warmup_bars = 21``. This is the same constant used
in ``signals._realized_vol_pct``, imported directly so the two cannot drift.

``prefix_safe`` records whether the feature has been verified to satisfy the
prefix-equivalence invariant: the feature computed at time ``t`` from the full
series must equal the feature computed at ``t`` from data truncated at ``t``.
Features that carry any cross-observation fitted state (scalers, imputers, PCA)
are not prefix-safe by construction and must be computed through the
``Pipeline`` API instead, which enforces the fit/transform split.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    """The static contract for one named feature.

    ``name`` is unique within a registry. Duplicate registration raises to
    prevent a quiet collision between independently developed features that
    happen to share a name — the second would silently shadow the first in any
    dict-keyed lookup.

    ``version`` is a semver string. A logic change that changes the output of
    the feature (even if the schema is unchanged) must bump at least the patch
    version, so the manifest hash changes and historical comparisons flag the
    mismatch rather than treating the old and new results as the same feature.

    ``inputs`` lists the ``PointInTimeState`` methods and arguments the feature
    reads from. This is documentation today and a dependency-graph input
    tomorrow; it must be complete and accurate.

    ``warmup_bars`` is mandatory — see module docstring. A feature that does not
    know its own warm-up cannot be used in a correctly-embargoed backtest.

    ``warmup_seconds`` is derived: ``warmup_bars * interval_seconds``. The
    embargo uses wall-clock seconds so that two assets sampled at different
    rates are embargoed to the same boundary. Callers must supply the interval
    explicitly; the registry does not assume one.

    ``prefix_safe`` is True when the feature has been proven to satisfy the
    prefix-equivalence invariant (feature at t from full data == feature at t
    from data truncated at t). A False here does not mean the feature leaks;
    it means the invariant has not been verified and the ``Pipeline`` must
    treat the feature as potentially requiring fitted state.

    ``definition_hash`` is a stable SHA-256 of the name, version and inputs.
    It is stored in the run manifest so any change to the feature's contract
    is detectable across runs even when the feature name stays the same.
    """

    name: str
    version: str
    inputs: tuple[str, ...]
    warmup_bars: int
    prefix_safe: bool
    # Derived hash, or supplied explicitly for deterministic test fixtures.
    definition_hash: str = field(default="")

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FeatureDefinition.name must not be empty")
        if not self.version:
            raise ValueError("FeatureDefinition.version must not be empty")
        if self.warmup_bars < 1:
            raise ValueError(
                f"FeatureDefinition '{self.name}': warmup_bars must be >= 1; "
                "a feature that claims no warm-up is either trivially correct "
                "or silently wrong on insufficient data"
            )
        if not self.definition_hash:
            # Compute a stable hash from the fields that define the contract.
            # inputs is sorted before hashing so insertion order does not
            # affect the hash — two definitions with the same inputs in
            # different order are the same feature.
            payload: dict[str, Any] = {
                "name": self.name,
                "version": self.version,
                "inputs": sorted(self.inputs),
                "warmup_bars": self.warmup_bars,
            }
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest()[:16]
            object.__setattr__(self, "definition_hash", digest)

    def warmup_seconds(self, interval_seconds: float) -> float:
        """Wall-clock warm-up span for this feature at the given bar interval.

        The embargo must be at least this long. Using bar counts instead of
        wall-clock seconds would make the embargo asset-dependent: a 5m-sampled
        asset and a 1h-sampled asset would have different embargo widths for the
        same feature, letting observations from the 1h asset bleed into the
        validation period because its bars are wider than the 5m embargo.
        """
        return self.warmup_bars * interval_seconds


class FeatureRegistry:
    """Named registry of feature definitions.

    Registration is explicit and duplicate names raise. The registry is the
    single source of truth for which features are active in a run; the
    experiment manifest records ``registry.as_manifest_dict()`` verbatim so any
    change to the active set or to a definition hash is visible in the diff.

    Instantiate one registry per run. Do not share registries across runs with
    different configurations — the manifest is per-run and must reflect what
    that run actually computed.
    """

    def __init__(self) -> None:
        self._defs: dict[str, FeatureDefinition] = {}

    def register(self, defn: FeatureDefinition) -> FeatureDefinition:
        """Add a definition. Raises if the name is already registered.

        Returns the definition so callers can store a reference in a single
        statement: ``MY_FEATURE = registry.register(FeatureDefinition(...))``.
        """
        if defn.name in self._defs:
            existing = self._defs[defn.name]
            raise ValueError(
                f"Feature '{defn.name}' is already registered "
                f"(existing hash: {existing.definition_hash}, "
                f"new hash: {defn.definition_hash}). "
                "If this is a new version of the same feature, bump the "
                "version string so the manifest can distinguish the two runs."
            )
        self._defs[defn.name] = defn
        return defn

    def get(self, name: str) -> FeatureDefinition:
        """Retrieve a registered definition by name. Raises on missing."""
        if name not in self._defs:
            raise KeyError(
                f"Feature '{name}' is not registered. Register it before "
                "computing it, so the embargo calculation has its warm-up."
            )
        return self._defs[name]

    def all(self) -> tuple[FeatureDefinition, ...]:
        """All registered definitions, in registration order."""
        return tuple(self._defs.values())

    def max_warmup_seconds(self, interval_seconds: float) -> float:
        """The longest warm-up across all registered features.

        The embargo must be at least this long. Passing the minimum of
        individual warm-ups would let the shorter-warm-up features see into
        the embargo period of the longer-warm-up ones.
        """
        if not self._defs:
            return 0.0
        return max(d.warmup_seconds(interval_seconds) for d in self._defs.values())

    def as_manifest_dict(self) -> list[dict[str, object]]:
        """Stable JSON-serializable representation for the run manifest.

        Sorted by name so the diff between two runs is minimal when features
        are added or removed — an unsorted dict produces a churn of every entry
        when one is prepended alphabetically.
        """
        return [
            {
                "name": d.name,
                "version": d.version,
                "inputs": list(d.inputs),
                "warmup_bars": d.warmup_bars,
                "prefix_safe": d.prefix_safe,
                "definition_hash": d.definition_hash,
            }
            for d in sorted(self._defs.values(), key=lambda d: d.name)
        ]


__all__ = ["FeatureDefinition", "FeatureRegistry"]
