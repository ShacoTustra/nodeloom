"""Model providers: who runs the models the nodes call.

A provider is a base URL, a key and a dialect (its `type`):

- openai: the OpenAI API and what speaks it (OpenAI, OpenRouter, Groq, DeepSeek, Ollama, LM Studio, vLLM, ...): text
  through /chat/completions, GPT Image pictures through /images/generations and /images/edits;
- google: the Gemini API: text through its OpenAI-compatible /chat/completions, Nano Banana pictures and Omni video
  through the Interactions API;
- ark: BytePlus ModelArk, Volcengine Ark and gateways in front of them: text through /chat/completions, Seedream
  pictures, Seedance video tasks, and ready characters for Seedance where the gateway lists them (/assets);
- fal: fal.ai's queue; every endpoint is a node of its own, made of the endpoint's OpenAPI schema;
- replicate: Replicate's predictions; every model is a node of its own, made of its version's OpenAPI schema.

In the nodes a model is "<provider>/<model id>". What kind it is (chat, image, video) and what a chat model takes in
besides text come from the provider's GET /models, completed or replaced by the config. Each node lists the models
of its family: the text model node every chat model, the Seedream node the image models of ark providers, and so on.
"""

from __future__ import annotations

import asyncio
import logging
import re

from nodeloom.config import ProviderSettings, Settings, get_settings
from nodeloom.providers.base import FAMILIES, Model, Provider, ProviderError

log = logging.getLogger(__name__)

__all__ = ["FAMILIES", "Model", "Provider", "ProviderError", "all_models", "choices", "configured", "get", "registry",
           "resolve"]

NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
OPENROUTER = "https://openrouter.ai/api/v1"

_registry: tuple[Settings, dict[str, Provider]] | None = None


def configured(s: Settings) -> dict[str, ProviderSettings]:
    """The providers of the config file, and the common ones from their environment variables alone."""
    out = dict(s.providers)
    if s.openai_api_key and "openai" not in out:
        out["openai"] = ProviderSettings(type="openai", base_url=s.openai_base_url, api_key=s.openai_api_key)
    if s.openrouter_api_key and "openrouter" not in out:
        out["openrouter"] = ProviderSettings(type="openai", base_url=OPENROUTER, api_key=s.openrouter_api_key)
    if (s.gemini_api_key or s.google_api_key) and "google" not in out:
        out["google"] = ProviderSettings(type="google", api_key=s.gemini_api_key or s.google_api_key)
    if s.fal_key and "fal" not in out:
        out["fal"] = ProviderSettings(type="fal", api_key=s.fal_key)
    if s.replicate_api_token and "replicate" not in out:
        out["replicate"] = ProviderSettings(type="replicate", api_key=s.replicate_api_token)
    if s.ark_api_key and "ark" not in out:
        out["ark"] = ProviderSettings(type="ark", base_url=s.ark_base_url, api_key=s.ark_api_key)
    return out


def _build(name: str, ps: ProviderSettings) -> Provider:
    from nodeloom.providers.ark import ArkProvider
    from nodeloom.providers.fal import FalProvider
    from nodeloom.providers.google import GoogleProvider
    from nodeloom.providers.openai import OpenAIProvider
    from nodeloom.providers.replicate import ReplicateProvider

    types = {"openai": OpenAIProvider, "google": GoogleProvider, "ark": ArkProvider, "fal": FalProvider,
             "replicate": ReplicateProvider}
    if not NAME.match(name):
        raise ValueError(f"provider name {name!r}: lowercase letters, digits, - and _ only")
    if ps.type not in types:
        raise ValueError(f"provider {name}: unknown type {ps.type!r} (one of: {', '.join(types)})")
    return types[ps.type](name, ps)


def registry() -> dict[str, Provider]:
    """The configured providers by name (built again when the settings change, as in tests)."""
    global _registry
    s = get_settings()
    if _registry is None or _registry[0] is not s:
        _registry = (s, {name: _build(name, ps) for name, ps in configured(s).items()})
    return _registry[1]


def get(name: str) -> Provider | None:
    return registry().get(name)


_nodes_lock = asyncio.Lock()


async def ensure_nodes() -> None:
    """The nodes the providers make of their APIs' schemas (fal, Replicate), registered with the static ones (each
    provider keeps them for a few minutes, so this is cheap)."""
    from nodeloom import nodes

    async with _nodes_lock:
        made = []
        for p in registry().values():
            if hasattr(p, "nodes"):
                made += await p.nodes()
        nodes.set_dynamic(made)


async def all_models() -> list[Model]:
    out: list[Model] = []
    for p in registry().values():
        out += await p.models()
    return out


async def choices() -> dict[str, list[str]]:
    """The model lists of the nodes: {"models:<family>": ["<provider>/<model>", ...]} for every family."""
    out: dict[str, list[str]] = {f"models:{f}": [] for f in FAMILIES}
    for m in await all_models():
        out[f"models:{m.family}"].append(m.value)
    return out


async def resolve(value: str) -> tuple[Provider, Model]:
    """The provider and the model of a node's model value; ProviderError (a message for the user) when there is none."""
    from nodeloom import i18n

    name, _, model_id = (value or "").partition("/")
    p = get(name)
    if p is None or not model_id:
        raise ProviderError(i18n.tr(f"Провайдер «{name}» не настроен: выберите другую модель",
                                    f"Provider “{name}” is not configured: choose another model"))
    for m in await p.models():
        if m.id == model_id:
            return p, m
    raise ProviderError(i18n.tr(f"У провайдера «{name}» нет модели {model_id}",
                                f"Provider “{name}” has no model {model_id}"))


async def close() -> None:
    for p in registry().values():
        await p.close()
