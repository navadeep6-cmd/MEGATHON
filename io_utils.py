"""Small file helpers: JSON / JSONL with atomic writes."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, Iterable, Iterator, List


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _atomic_write_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def write_json(path: str, obj: Any, indent: int = 2) -> None:
    _atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent))


def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    _atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def append_jsonl(path: str, row: dict) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl(path: str) -> List[dict]:
    return list(iter_jsonl(path))
