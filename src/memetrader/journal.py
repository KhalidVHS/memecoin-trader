"""Append-only JSONL logs, and the one place that knows how to serialize our
dataclasses.

Both logs are append-only on purpose. A trading ledger you can rewrite is a
trading ledger you cannot learn from — the whole value of this project is being
able to go back and ask why a losing trade looked like a good idea at the time.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses, pydantic models, enums and tuples into
    plain JSON types. Enums are StrEnum so they serialize as their value."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float):
        # inf and nan are legal Python but not legal JSON. TxnCounts.ratio
        # returns inf for a pool with zero sells, which is meaningful — encode
        # it as a string so it survives the round trip visibly.
        if obj != obj:  # NaN
            return None
        if obj in (float("inf"), float("-inf")):
            return str(obj)
        return obj
    return obj


def append(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")


def read(path: Path) -> Iterator[dict]:
    """Yield every row. A corrupt trailing line (a crash mid-write) is skipped
    rather than fatal — the rest of the ledger is still good."""
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def tail(path: Path, n: int) -> list[dict]:
    rows = list(read(path))
    return rows[-n:] if n > 0 else []
