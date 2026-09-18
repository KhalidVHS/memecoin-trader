"""A paper broker that charges you for everything a real Solana swap charges.

The whole point of this module is that the P&L it reports is *believable*. Three
costs are what separate an honest paper run from a fantasy one, and all three
are modelled here:

* **The pool fee** — but *only on the degraded path*. See ``pool_fee_rate``:
  a real Jupiter quote already has every hop's fee deducted inside it, so
  charging one again would double-count.
* **Gas**, charged on *every* attempt — including the ones that fail.
* **Failed transactions.** Solana swaps fail routinely: the route goes stale,
  slippage is exceeded, the block is full. A naive simulator fills 100% of
  orders for free. Here a configured fraction of attempts fail, and a failed
  attempt still burns gas. Over a few hundred trades that difference is the
  difference between a strategy that looks profitable and one that is.

Cost basis is recorded **including fees and gas**, which is what the user
actually paid. That makes a position show a small loss the instant it is
filled, which is correct and which no amount of optimism should paper over.

State is one JSON file, written atomically (temp file plus ``os.replace``), so
a crash or a Ctrl+C mid-write can never leave a truncated ledger. Every attempt,
successful or not, appends one row to ``trades.jsonl``.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

from .config import Config
from .types import Fill, FillQuote, Position, Side

#: Bumped whenever the on-disk shape changes. A state file written by a newer
#: version is refused rather than silently misread.
SCHEMA_VERSION = 1

#: Relative tolerance for "this position is now empty". Absolute epsilons are
#: useless here: a BONK position is ~10^7 tokens and a WIF position is ~10^2,
#: so the same absolute dust threshold cannot be right for both.
_QTY_REL_EPSILON = 1e-9


def pool_fee_rate(quote: FillQuote) -> float:
    """The pool fee to charge *on top of* ``quote.price_usd``, as a fraction.

    **Zero for a real Jupiter route, and that is not a bug.** Jupiter's
    ``outAmount`` is the number of tokens the pools actually send you, so every
    hop's AMM fee — along with slippage and price impact — is already deducted
    *inside* the route. ``quote.price_usd`` is ``outAmount/inAmount``, which
    means the fee is already in the price. Charging ``pool_fee_pct`` again
    would bill the user twice. The live evidence is unambiguous: BONK quoted at
    2.9636e-6 to buy and 2.9623e-6 to sell, a 4.4 bp round trip, where a 2-hop
    route billed at 0.25% per hop would imply 200 bp — 45x the observed spread.

    The degraded path is different. There ``quotes.py`` could not reach Jupiter
    and synthesised a price from a DexScreener mid plus a slippage assumption.
    A mid is a raw price with no fee baked into it, so on that path the pool
    fee is a real, uncounted cost and is charged explicitly.
    """
    if not quote.degraded:
        return 0.0
    return float(quote.pool_fee_pct) / 100.0  # pool_fee_pct is a whole percent


class BrokerError(RuntimeError):
    """Base class for everything this module refuses to do."""


class InsufficientCash(BrokerError):
    """A BUY was larger than the cash available, fees and gas included."""


class NoPosition(BrokerError):
    """A SELL was placed for a symbol with no open position.

    ``risk.check`` catches this first in the normal flow; reaching the broker
    means something upstream is out of sync, which is worth a loud failure
    rather than a silent no-op fill.
    """


class LocalPaperBroker:
    """The ``Broker`` protocol, backed by a JSON file on disk.

    ``place_order`` takes a keyword-only ``quote`` beyond the protocol's two
    positional parameters. The protocol is the minimum surface a venue must
    offer; a paper venue needs to be told what price it would have got.
    """

    def __init__(self, cfg: Config, *, rng: random.Random | None = None) -> None:
        self.cfg = cfg
        # Injectable so tests are deterministic. The default is seeded from the
        # OS, because a reproducible live run would be a lie of a different kind.
        self._rng = rng if rng is not None else random.Random()

        self._cash_usd: float = float(cfg.starting_cash_usd)
        self._positions: dict[str, Position] = {}
        self._realized_pnl_usd: float = 0.0
        self._fees_paid_usd: float = 0.0
        self._gas_paid_usd: float = 0.0
        self._starting_cash_usd: float = float(cfg.starting_cash_usd)

        self.load()

    # -- read-only view ----------------------------------------------------

    @property
    def cash_usd(self) -> float:
        return self._cash_usd

    @property
    def realized_pnl_usd(self) -> float:
        """Cumulative since inception, net of the fees and gas of each exit."""
        return self._realized_pnl_usd

    @property
    def fees_paid_usd(self) -> float:
        return self._fees_paid_usd

    @property
    def gas_paid_usd(self) -> float:
        """Includes gas burned on failed transactions."""
        return self._gas_paid_usd

    @property
    def starting_cash_usd(self) -> float:
        """From the state file once it exists, so editing ``config.toml``
        mid-run cannot retroactively rewrite the return number."""
        return self._starting_cash_usd

    def get_positions(self) -> dict[str, Position]:
        return dict(self._positions)

    # -- orders ------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        side: Side,
        usd_notional: float,
        *,
        quote: FillQuote,
        now: float | None = None,
    ) -> Fill:
        """Attempt one swap. Always returns a ``Fill``; never returns ``None``.

        Raises ``InsufficientCash`` or ``NoPosition`` when the order could not
        even be broadcast — those are upstream bugs, not market outcomes, and
        they cost nothing because no transaction was ever sent.
        """
        ts = time.time() if now is None else now
        side = Side(side)

        if not usd_notional > 0:
            raise ValueError(f"usd_notional must be positive, got {usd_notional!r}")
        if not quote.price_usd > 0:
            raise ValueError(f"quote.price_usd must be positive for {symbol}")

        gas = float(self.cfg.execution.gas_usd_per_swap)
        fee_rate = pool_fee_rate(quote)

        # --- pre-flight: can this transaction be sent at all? --------------
        if side is Side.BUY:
            required = usd_notional * (1.0 + fee_rate) + gas
            if self._cash_usd < required:
                raise InsufficientCash(
                    f"BUY {symbol} ${usd_notional:,.2f} needs ${required:,.2f} "
                    f"(notional + ${usd_notional * fee_rate:,.2f} pool fee + "
                    f"${gas:,.2f} gas) but cash is ${self._cash_usd:,.2f}"
                )
            filled_usd = usd_notional
            note: str | None = None
        else:
            position = self._positions.get(symbol)
            if position is None:
                raise NoPosition(f"SELL {symbol}: no open position")
            quantity_wanted = usd_notional / quote.price_usd
            if quantity_wanted > position.quantity:
                # Selling more than you own must clamp, never go short.
                filled_usd = position.quantity * quote.price_usd
                note = (
                    f"clamped: requested ${usd_notional:,.2f} exceeds the "
                    f"${filled_usd:,.2f} position"
                )
            else:
                filled_usd = usd_notional
                note = None

        # --- the chain's coin flip ----------------------------------------
        # Drawn once per attempt, after validation, so a seeded rng replays a
        # run exactly. A failed swap pays full gas and changes nothing else.
        if self._rng.random() < self.cfg.execution.failed_tx_rate:
            self._cash_usd -= gas
            self._gas_paid_usd += gas
            fill = Fill(
                ts=ts,
                symbol=symbol,
                side=side,
                requested_usd=usd_notional,
                filled_usd=0.0,
                price_usd=quote.price_usd,
                quantity=0.0,
                price_impact_pct=quote.price_impact_pct,
                pool_fee_usd=0.0,  # the pool never executed; only gas was spent
                gas_usd=gas,
                realized_pnl_usd=0.0,
                failed=True,
                degraded=quote.degraded,
                note="transaction failed; gas charged",
            )
            self._commit(fill)
            return fill

        pool_fee = filled_usd * fee_rate
        quantity = filled_usd / quote.price_usd
        realized = 0.0

        if side is Side.BUY:
            cost = filled_usd + pool_fee + gas
            self._cash_usd -= cost
            existing = self._positions.get(symbol)
            if existing is None:
                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_entry_price_usd=quote.price_usd,
                    opened_at=ts,
                    cost_basis_usd=cost,
                )
            else:
                total_quantity = existing.quantity + quantity
                # Notional-weighted average *price*; the fee and gas drag lives
                # in cost_basis_usd, which is what P&L is measured against.
                notional = (
                    existing.quantity * existing.avg_entry_price_usd
                    + quantity * quote.price_usd
                )
                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=total_quantity,
                    avg_entry_price_usd=notional / total_quantity,
                    opened_at=existing.opened_at,  # age is from the first entry
                    cost_basis_usd=existing.cost_basis_usd + cost,
                )
        else:
            position = self._positions[symbol]
            if quantity >= position.quantity:
                # Includes the clamped case, where these are equal by
                # construction. Pinning the fraction at exactly 1.0 avoids
                # leaving 10^-16 of a token behind.
                quantity = position.quantity
                fraction = 1.0
            else:
                fraction = quantity / position.quantity

            basis_share = position.cost_basis_usd * fraction
            realized = filled_usd - basis_share - pool_fee - gas
            self._cash_usd += filled_usd - pool_fee - gas
            self._realized_pnl_usd += realized

            remaining_quantity = position.quantity - quantity
            if remaining_quantity <= position.quantity * _QTY_REL_EPSILON:
                del self._positions[symbol]
            else:
                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=remaining_quantity,
                    avg_entry_price_usd=position.avg_entry_price_usd,
                    opened_at=position.opened_at,
                    cost_basis_usd=position.cost_basis_usd - basis_share,
                )

        self._fees_paid_usd += pool_fee
        self._gas_paid_usd += gas

        fill = Fill(
            ts=ts,
            symbol=symbol,
            side=side,
            requested_usd=usd_notional,
            filled_usd=filled_usd,
            price_usd=quote.price_usd,
            quantity=quantity,
            price_impact_pct=quote.price_impact_pct,
            pool_fee_usd=pool_fee,
            gas_usd=gas,
            realized_pnl_usd=realized,
            failed=False,
            degraded=quote.degraded,
            note=note,
        )
        self._commit(fill)
        return fill

    # -- persistence -------------------------------------------------------

    def load(self) -> None:
        """Read the state file. A missing file is a cold start, not an error.

        A corrupt or future-versioned file *is* an error: resetting to starting
        cash would quietly erase the entire P&L history, which is the worst
        possible failure mode for a ledger.
        """
        path = self.cfg.state_path
        if not path.exists():
            self._cash_usd = float(self.cfg.starting_cash_usd)
            self._starting_cash_usd = float(self.cfg.starting_cash_usd)
            self._positions = {}
            self._realized_pnl_usd = 0.0
            self._fees_paid_usd = 0.0
            self._gas_paid_usd = 0.0
            return

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise BrokerError(
                f"{path} is corrupt and will not be overwritten: {exc}. "
                f"Move it aside to start fresh."
            ) from exc

        version = int(payload.get("schema_version", 0))
        if version > SCHEMA_VERSION:
            raise BrokerError(
                f"{path} has schema_version {version}, but this build "
                f"understands at most {SCHEMA_VERSION}"
            )

        self._cash_usd = float(payload["cash_usd"])
        self._realized_pnl_usd = float(payload.get("realized_pnl_usd", 0.0))
        self._fees_paid_usd = float(payload.get("fees_paid_usd", 0.0))
        self._gas_paid_usd = float(payload.get("gas_paid_usd", 0.0))
        self._starting_cash_usd = float(
            payload.get("starting_cash_usd", self.cfg.starting_cash_usd)
        )
        self._positions = {
            symbol: Position(
                symbol=symbol,
                quantity=float(raw["quantity"]),
                avg_entry_price_usd=float(raw["avg_entry_price_usd"]),
                opened_at=float(raw["opened_at"]),
                cost_basis_usd=float(raw["cost_basis_usd"]),
            )
            for symbol, raw in (payload.get("positions") or {}).items()
        }

    def save(self) -> None:
        """Serialize state atomically: temp file in the same directory, fsync,
        then ``os.replace``, which is atomic on both POSIX and Windows."""
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "cash_usd": self._cash_usd,
            "starting_cash_usd": self._starting_cash_usd,
            "realized_pnl_usd": self._realized_pnl_usd,
            "fees_paid_usd": self._fees_paid_usd,
            "gas_paid_usd": self._gas_paid_usd,
            "positions": {
                symbol: {
                    "quantity": p.quantity,
                    "avg_entry_price_usd": p.avg_entry_price_usd,
                    "opened_at": p.opened_at,
                    "cost_basis_usd": p.cost_basis_usd,
                }
                for symbol, p in sorted(self._positions.items())
            },
        }
        _atomic_write_text(
            self.cfg.state_path,
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )

    # -- internals ---------------------------------------------------------

    def _commit(self, fill: Fill) -> None:
        """Append the fill to the trade log, then persist state.

        Log first: a trade that happened but was not saved is recoverable from
        the log, while a saved balance with no corresponding row is not
        explainable to the user.
        """
        self._append_trade(fill)
        self.save()

    def _append_trade(self, fill: Fill) -> None:
        row = {
            "ts": fill.ts,
            "symbol": fill.symbol,
            "side": str(fill.side),
            "requested_usd": fill.requested_usd,
            "filled_usd": fill.filled_usd,
            "price_usd": fill.price_usd,
            "quantity": fill.quantity,
            "price_impact_pct": fill.price_impact_pct,
            "pool_fee_usd": fill.pool_fee_usd,
            "gas_usd": fill.gas_usd,
            "realized_pnl_usd": fill.realized_pnl_usd,
            "failed": fill.failed,
            "degraded": fill.degraded,
            "note": fill.note,
        }
        path = self.cfg.trades_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        # Includes KeyboardInterrupt: leave no debris behind on Ctrl+C.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


__all__ = [
    "SCHEMA_VERSION",
    "BrokerError",
    "InsufficientCash",
    "LocalPaperBroker",
    "NoPosition",
    "pool_fee_rate",
]
