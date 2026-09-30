"""Configuracion central del gemelo.

Todas las rutas cuelgan de una unica raiz para que `dvc repro` pueda regenerar
el arbol completo desde cero (criterio de exito minimo #1 del plan, §16).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="XDT_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    data_root: Path = Field(default=REPO_ROOT / "data")

    # --- Credenciales / cortesia de red -------------------------------------
    epht_token: str | None = None
    user_agent: str = "xai-dt-heat/0.1 (research)"

    # --- Politica de reintentos (§14 R6b: 429 observado sin token) -----------
    http_max_retries: int = 5
    http_backoff_base_s: float = 5.0  # el plan fija el retroceso exponencial desde 5 s
    http_backoff_max_s: float = 300.0
    http_timeout_s: float = 60.0

    # Si es True, un conector nunca sale a la red: solo lee de data/raw.
    # Es el modo que usa `dvc repro` para garantizar reproducibilidad bit a bit.
    offline: bool = False

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_root / "interim"

    @property
    def features_dir(self) -> Path:
        return self.data_root / "features"

    @property
    def models_dir(self) -> Path:
        return self.data_root / "models"

    @property
    def twin_db_path(self) -> Path:
        return self.data_root / "twin.duckdb"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.interim_dir, self.features_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
