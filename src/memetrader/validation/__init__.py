"""Validation layer — anti-overfitting infrastructure for the backtest.

Sibling modules (added by independent agents):
  splits          — interval-aware purged cross-validation splitters
  walk_forward    — fit/select/refit/test cycle driver
  bootstrap       — stationary bootstrap for SPA/RC tests
  multiple_testing — p-value corrections and familywise error control
  leakage         — static and runtime leakage detectors
  recursive       — recursive / combinatorial feature importance
  ablation        — locked ablation experiment infrastructure
  promotion       — gating rules and FidelityTier enforcement

Do not add imports here; each sibling is imported by name where needed so that
circular dependencies between the siblings do not arise and so that agents
writing each module cannot accidentally depend on one another's not-yet-written
interfaces.
"""
