"""Cliente HTTP con cache inmutable y retroceso exponencial.

Dos riesgos del plan se atienden aqui:

  R6  · un endpoint cambia sin aviso y rompe el pipeline
        -> todo lo descargado queda en data/raw, inmutable y direccionado por
           contenido. Si manana el endpoint muere, el pipeline sigue corriendo.

  R6b · limite de tasa de la API bloquea la ingesta
        -> 429 observado repetidamente sin token. Retroceso exponencial desde
           5 s, respetando Retry-After, y cache agresiva para no repetir
           ninguna peticion ya servida.

El modo `offline` (settings.offline) prohibe salir a la red: solo lee cache.
Es el modo con el que debe correr `dvc repro` para que la reproducibilidad sea
real y no dependa de que un servidor externo siga en pie.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from xdt.config import Settings, get_settings
from xdt.hashing import hash_bytes, hash_json

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class OfflineCacheMiss(RuntimeError):
    """Se pidio un recurso no cacheado estando en modo offline."""


class RetryableBody(RuntimeError):
    """El cuerpo indica un fallo transitorio pese a un codigo HTTP de exito.

    Algunos servicios senalan el limite de tasa DENTRO de una respuesta 200.
    EPHT lo hace: devuelve HTTP 200 con {"code":429,"status":"Too Many
    Requests"}. Sin esta comprobacion, el cliente cachearia el documento de
    error como si fuera dato y el parser produciria silenciosamente cero
    filas.
    """


class InvalidBody(RuntimeError):
    """El cuerpo es un error permanente: no reintentar, no cachear."""


@dataclass(frozen=True)
class RawArtifact:
    """Una descarga cruda, con su procedencia."""

    raw_hash: str  # sha256 del contenido
    request_key: str  # sha256 de (source, resource, params)
    source: str
    resource: str
    params: dict[str, Any]
    path: Path
    n_bytes: int
    content_type: str
    fetched_at: dt.datetime
    from_cache: bool

    def bytes(self) -> bytes:
        return self.path.read_bytes()

    def text(self, encoding: str = "utf-8") -> str:
        return self.path.read_text(encoding=encoding, errors="replace")

    def json(self) -> Any:
        return json.loads(self.text())


class CachedClient:
    """Cliente HTTP idempotente con cache en data/raw."""

    def __init__(self, source: str, settings: Settings | None = None,
                 base_url: str = "", headers: dict[str, str] | None = None,
                 body_check: Callable[[bytes], None] | None = None):
        self.source = source
        self.settings = settings or get_settings()
        self.base_url = base_url.rstrip("/")
        self.headers = {"User-Agent": self.settings.user_agent, **(headers or {})}
        # Validador opcional del cuerpo. Debe lanzar RetryableBody o
        # InvalidBody. Se ejecuta ANTES de escribir en cache, para que un
        # documento de error nunca acabe en data/raw.
        self.body_check = body_check
        self._client: httpx.Client | None = None

    # ----------------------------------------------------------------- cache
    def _cache_paths(self, request_key: str) -> tuple[Path, Path]:
        d = self.settings.raw_dir / self.source / request_key[:2]
        return d / f"{request_key}.bin", d / f"{request_key}.meta.json"

    def _read_cache(self, request_key: str) -> RawArtifact | None:
        blob, meta = self._cache_paths(request_key)
        if not (blob.exists() and meta.exists()):
            return None
        m = json.loads(meta.read_text(encoding="utf-8"))
        return RawArtifact(
            raw_hash=m["raw_hash"],
            request_key=request_key,
            source=m["source"],
            resource=m["resource"],
            params=m.get("params", {}),
            path=blob,
            n_bytes=m["n_bytes"],
            content_type=m.get("content_type", ""),
            fetched_at=dt.datetime.fromisoformat(m["fetched_at"]),
            from_cache=True,
        )

    def _cache_entry_is_valid(self, art: RawArtifact) -> bool:
        """Revalida una entrada de cache antes de servirla.

        Una entrada escrita por una version anterior del cliente puede
        contener un sobre de error (p.ej. el 429 que EPHT devuelve dentro de
        un HTTP 200). Servirla indefinidamente propagaria el fallo para
        siempre; tratarla como fallo de cache la deja auto-repararse en la
        siguiente descarga.
        """
        if self.body_check is None:
            return True
        try:
            self.body_check(art.bytes())
        except (RetryableBody, InvalidBody):
            return False
        return True

    def _write_cache(
        self, request_key: str, resource: str, params: dict[str, Any],
        content: bytes, content_type: str,
    ) -> RawArtifact:
        blob, meta = self._cache_paths(request_key)
        blob.parent.mkdir(parents=True, exist_ok=True)
        raw_hash = hash_bytes(content)
        fetched_at = dt.datetime.now(dt.UTC).replace(tzinfo=None)
        # Escritura atomica: nunca dejar un .bin a medias si el proceso muere.
        tmp = blob.with_suffix(".bin.tmp")
        tmp.write_bytes(content)
        tmp.replace(blob)
        meta.write_text(
            json.dumps(
                {
                    "raw_hash": raw_hash,
                    "source": self.source,
                    "resource": resource,
                    "params": params,
                    "n_bytes": len(content),
                    "content_type": content_type,
                    "fetched_at": fetched_at.isoformat(),
                },
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )
        return RawArtifact(
            raw_hash=raw_hash,
            request_key=request_key,
            source=self.source,
            resource=resource,
            params=params,
            path=blob,
            n_bytes=len(content),
            content_type=content_type,
            fetched_at=fetched_at,
            from_cache=False,
        )

    # ------------------------------------------------------------------- net
    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                headers=self.headers,
                timeout=self.settings.http_timeout_s,
                follow_redirects=True,
            )
        return self._client

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), self.settings.http_backoff_max_s)
            except ValueError:
                pass
        return min(
            self.settings.http_backoff_base_s * (2**attempt),
            self.settings.http_backoff_max_s,
        )

    def get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        resource: str | None = None,
        refresh: bool = False,
    ) -> RawArtifact:
        """GET con cache. Devuelve siempre un RawArtifact."""
        params = params or {}
        url = f"{self.base_url}/{path.lstrip('/')}" if self.base_url else path
        logical = resource or url
        request_key = hash_json({"source": self.source, "resource": logical, "params": params})

        if not refresh:
            cached = self._read_cache(request_key)
            if cached is not None and self._cache_entry_is_valid(cached):
                return cached

        if self.settings.offline:
            raise OfflineCacheMiss(
                f"modo offline y sin cache para {logical} {params}. "
                "Ejecuta la ingesta en linea una vez, o ajusta XDT_OFFLINE=0."
            )

        last_exc: Exception | None = None
        for attempt in range(self.settings.http_max_retries):
            try:
                resp = self.client.get(url, params=params or None)
            except httpx.TransportError as exc:  # red caida, DNS, timeout
                last_exc = exc
                time.sleep(self._backoff(attempt, None))
                continue

            if resp.status_code in RETRYABLE_STATUS:
                last_exc = httpx.HTTPStatusError(
                    f"{resp.status_code} en {url}", request=resp.request, response=resp
                )
                if attempt < self.settings.http_max_retries - 1:
                    time.sleep(self._backoff(attempt, resp.headers.get("Retry-After")))
                    continue
                raise last_exc

            resp.raise_for_status()

            if self.body_check is not None:
                try:
                    self.body_check(resp.content)
                except RetryableBody as exc:
                    last_exc = exc
                    if attempt < self.settings.http_max_retries - 1:
                        time.sleep(self._backoff(attempt, resp.headers.get("Retry-After")))
                        continue
                    raise

            return self._write_cache(
                request_key,
                logical,
                params,
                resp.content,
                resp.headers.get("Content-Type", ""),
            )

        raise last_exc or RuntimeError(f"fallo irrecuperable en {url}")

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> CachedClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
