"""Deterministic replay engine for the memecoin trader.

This package contains the three primitives that every other backtest module
builds on: a simulated clock, a deterministic event queue, and the run config
that governs them.  They are separated from the engine itself (``engine.py``,
not yet written) so they can be tested offline without touching any live module.

Import order within the package deliberately avoids circular imports:
``types`` → ``clock`` → ``event_queue`` → ``config``.
"""

from __future__ import annotations
