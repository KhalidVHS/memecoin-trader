"""``memetrader`` — an LLM-assisted Solana memecoin **paper** trader.

There is no wallet, no private key and no signing path in this package. Every
"fill" is simulated against a local ledger by ``broker.LocalPaperBroker``, and
:class:`~memetrader.types.ExecutionMode` makes that structural rather than
customary: ``LIVE`` has no implementation and ``assert_live_supported`` exists
to stop one appearing by accident. The audit's C12 finding — ``--dry-run``
could still liquidate a position, because the stop path never consulted the
flag — is why the capability travels with the broker object instead of being a
boolean somebody remembers to check.

The package is deliberately a modular monolith with one synchronous writer.
Module ownership, in dependency order:

* ``types``     — the shared vocabulary. Every other module's boundary.
* ``ids``       — immutable, time-sortable identifiers (audit C11).
* ``http``      — the single HTTP client factory, plus the retry budget,
                  backoff and per-host circuit breaker every caller shares.
* ``journal``   — the durable append-only ledger and the only serializer.
* ``market``, ``quotes``, ``sentiment`` — vendor adapters.
* ``signals``, ``strategy``, ``brain``  — features and decisions.
* ``risk``, ``portfolio``, ``broker``   — bounds, marks and execution.
* ``loop``, ``cli``, ``report``         — orchestration and presentation.

This file used to hold a ``hello()`` returned by ``uv init``. The audit named
it (§15, ``__init__.py::hello``) and it is gone; nothing imported it. A package
``__init__`` deliberately exports nothing here — every consumer imports the
module it actually depends on, which is what keeps the dependency direction
above readable in the import lines themselves.
"""

from __future__ import annotations

__all__: list[str] = []
