"""Hashing de contenido y procedencia.

L0 (§6) exige que cada prediccion se persista con el hash de sus datos de
entrada. Para que ese hash sea util tiene que ser *estable*: el mismo
DataFrame debe producir el mismo digest en cualquier maquina y en cualquier
orden de filas.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_CHUNK = 1 << 20


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def hash_json(obj: Any) -> str:
    """Digest estable de una estructura JSON-serializable (claves ordenadas)."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hash_bytes(payload.encode("utf-8"))


def hash_frame(df: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    """Digest de un DataFrame invariante al orden de filas y columnas.

    Se usa para el `input_hash` del registro de predicciones: dos ejecuciones
    con los mismos datos de entrada deben producir el mismo hash aunque el
    pipeline haya reordenado las filas.
    """
    cols = sorted(columns) if columns is not None else sorted(map(str, df.columns))
    sub = df.loc[:, cols]
    # np.sort devuelve copia: el array de hash_pandas_object es de solo lectura.
    row_digests = np.sort(pd.util.hash_pandas_object(sub, index=False).to_numpy())
    h = hashlib.sha256()
    h.update(",".join(cols).encode("utf-8"))
    h.update(row_digests.tobytes())
    return h.hexdigest()


def stable_id(*parts: Any) -> str:
    """Identificador determinista a partir de componentes de una clave natural."""
    joined = "\x1f".join("" if p is None else str(p) for p in parts)
    return hash_bytes(joined.encode("utf-8"))[:32]


def manifest(**fields: Any) -> Mapping[str, Any]:
    """Manifiesto de procedencia serializable, con su propio hash incluido."""
    body = {k: v for k, v in sorted(fields.items())}
    return {**body, "manifest_hash": hash_json(body)}
