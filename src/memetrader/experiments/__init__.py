"""Experiment provenance and governance — manifest and registry.

Two modules, one purpose: make it hard to accidentally produce a backtest result
that cannot be traced, cannot be reproduced, or has consumed its holdout degrees
of freedom without recording the fact.

``manifest`` captures every axis on which two runs can differ — code, data,
features, model, config, seeds — and makes two runs provably identical or
provably not via ``manifest_hash``.

``registry`` enforces the research process rules: every trial is registered
(including abandoned ones), the holdout may be opened once, any revision after
opening is flagged, and LLMs are prevented from seeing holdout results.
"""

from __future__ import annotations

from memetrader.experiments.manifest import (
    Manifest,
    build_manifest,
    manifest_hash,
    read_manifest,
    write_manifest,
)
from memetrader.experiments.registry import (
    Actor,
    CertifyResult,
    HoldoutAccessError,
    PermissionError,
    Registry,
    RegistryRow,
    TrialRegistration,
)

__all__ = [
    "Actor",
    "CertifyResult",
    "HoldoutAccessError",
    "Manifest",
    "PermissionError",
    "Registry",
    "RegistryRow",
    "TrialRegistration",
    "build_manifest",
    "manifest_hash",
    "read_manifest",
    "write_manifest",
]
