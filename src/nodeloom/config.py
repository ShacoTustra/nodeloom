"""Settings: the environment (or .env), then a TOML file (`CONFIG_FILE`, else config.toml in the working directory or
in the data directory). The environment wins.

Providers are configured in the TOML file:

    [providers.openai]
    type = "openai"
    base_url = "https://api.openai.com/v1"
    api_key = "sk-..."            # or api_key_env = "OPENAI_API_KEY"
    models = ["gpt-5-mini"]       # optional: otherwise the provider's GET /models

or, for the common ones, with an environment variable alone: OPENAI_API_KEY, OPENROUTER_API_KEY, ARK_API_KEY
(see providers/__init__.py).
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

APP_NAME = "Nodeloom"
APP_ID = "nodeloom"  # the data directory, the default database and bucket names


def default_data_dir() -> Path:
    """Where the database and the files live unless DATA_DIR says otherwise: the user's data directory."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / APP_ID


class ModelSettings(BaseModel):
    """A model of a provider, when the provider's own list says too little (or nothing)."""

    kind: str | None = None  # chat | image | video; guessed from the id when not given
    inputs: list[str] | None = None  # what a chat model takes besides text: image, audio, video
    # Prices in US dollars, for the cost shown after a run: input / output per million tokens, image per picture
    # (see providers/pricing.py). Without a price the run shows no cost.
    price: dict[str, float] | None = None
    title: str | None = None


class ProviderSettings(BaseModel):
    type: str  # openai | ark
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None  # the key from this environment variable instead
    # Models: a list of ids, or a table {id: {kind, inputs, price}}. Not given: the provider's GET /models.
    models: list[str] | dict[str, ModelSettings] | None = None
    # More models on top of the provider's own list (or settings for some of them).
    extra_models: dict[str, ModelSettings] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    # Hosts the provider's result files may be downloaded from (video links); the type's own when not given.
    file_hosts: list[str] | None = None
    timeout_s: float = 600.0

    def key(self) -> str | None:
        if self.api_key:
            return self.api_key
        return os.environ.get(self.api_key_env) if self.api_key_env else None


def _config_file() -> Path | None:
    if value := os.environ.get("CONFIG_FILE"):
        return Path(value)
    for p in (Path("config.toml"), Path(os.environ.get("DATA_DIR") or default_data_dir()) / "config.toml"):
        if p.is_file():
            return p
    return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Field(default_factory=default_data_dir)
    # Not given: SQLite in the data directory. PostgreSQL: postgresql+asyncpg://... (pip install nodeloom[postgres])
    database_url: str | None = None

    # Shared state of several processes: Redis through Sentinel ("host:port,host:port", master name), or a single
    # REDIS_URL. Unset: run events stay within the process (one process is enough for one user or a small team).
    redis_sentinels: str | None = None
    redis_master: str = APP_ID
    redis_password: str | None = None
    redis_url: str | None = None

    # Who may open the editor: the Host header must be one of these (DNS rebinding: a page of another site that
    # resolves its name to 127.0.0.1 must not reach the editor). "*" lets any host in (behind your own proxy and auth).
    allowed_hosts: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["localhost", "127.0.0.1", "::1"])
    # The editor's language when the user has not chosen one: en or ru. Not given: the browser's.
    language: str | None = None

    # Runs wait in the database; each process runs at most RUNS_PER_PROCESS of them and GENERATIONS_PER_PROCESS
    # generations at once; a user has at most RUNS_PER_USER running and QUEUE_PER_USER queued and running together.
    runs_per_user: int = 2
    queue_per_user: int = 20
    runs_per_process: int = 200
    generations_per_process: int = 100
    # On stop (deploy, restart) generations in flight get this long to finish before their runs go back to the queue.
    stop_grace_s: float = 60.0

    # The Builder (agent.py): a chat model with function calling, "<provider>/<model>". Not given: the first chat model.
    agent_model: str | None = None

    # Files (uploads, results): a directory (the data directory's media/ when not given), or S3-compatible storage
    # when the endpoint is set (pip install nodeloom[s3]).
    media_dir: Path | None = None
    media_s3_endpoint: str | None = None
    media_s3_bucket: str = APP_ID
    media_s3_access_key: str | None = None
    media_s3_secret_key: str | None = None
    media_s3_region: str = "us-east-1"
    # Room for a user's files in megabytes; 0: no limit.
    storage_limit_mb: int = 0

    providers: dict[str, ProviderSettings] = Field(default_factory=dict)
    # Shortcuts for the common providers (no config file needed).
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openrouter_api_key: str | None = None
    ark_api_key: str | None = None
    ark_base_url: str = "https://ark.ap-southeast.bytepluses.com/api/v3"

    @classmethod
    def settings_customise_sources(cls, settings_cls: type[BaseSettings], init_settings: PydanticBaseSettingsSource,
                                   env_settings: PydanticBaseSettingsSource, dotenv_settings: PydanticBaseSettingsSource,
                                   file_secret_settings: PydanticBaseSettingsSource):
        sources: list[Any] = [init_settings, env_settings, dotenv_settings, file_secret_settings]
        if (path := _config_file()) is not None:
            sources.append(TomlConfigSettingsSource(settings_cls, toml_file=path))
        return tuple(sources)

    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def _hosts(cls, v):
        return [h.strip() for h in v.split(",") if h.strip()] if isinstance(v, str) else v  # ALLOWED_HOSTS=a,b

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite+aiosqlite:///{self.data_dir / 'app.db'}"

    @property
    def media_path(self) -> Path:
        return self.media_dir or self.data_dir / "media"


@lru_cache
def get_settings() -> Settings:
    return Settings()
