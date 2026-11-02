"""What a call cost, in micro-dollars, from the usage the provider reported and the model's price.

Prices are US dollars: `input`, `output` and `image_input` per million tokens, `image` per generated picture,
`request` per call (fal's and Replicate's nodes), `second` per second of the model's run time (Replicate's community
models, priced by their hardware). Since a micro-dollar is a millionth of a dollar, tokens times dollars per million
tokens is already micro-dollars.
"""

from __future__ import annotations


def _int(v) -> int:
    try:
        return max(int(v or 0), 0)
    except (TypeError, ValueError):
        return 0


def _float(v) -> float:
    try:
        return max(float(v or 0), 0.0)
    except (TypeError, ValueError):
        return 0.0


def cost_micro(price: dict | None, usage: dict | None) -> int | None:
    """None when the cost is not known: no price for the model and no cost in the usage.

    - usage.cost (OpenRouter: dollars) is taken as it is;
    - chat: prompt_tokens / completion_tokens (or input_tokens / output_tokens) at `input` / `output`;
    - GPT Image: input_tokens_details.image_tokens at `image_input`, the rest of the input at `input`, output at
      `output`;
    - Seedream: generated_images at `image` (or output_tokens at `output`); Seedance: completion_tokens at `output`;
    - Gemini: input_tokens / output_tokens (thinking included) at `input` / `output`;
    - fal, Replicate: requests at `request`, generated_images at `image`, seconds (run time) at `second`.
    """
    usage = usage if isinstance(usage, dict) else {}
    if isinstance(usage.get("cost"), (int, float)) and not isinstance(usage.get("cost"), bool):
        return round(usage["cost"] * 1_000_000)
    if not price or not usage:
        return None
    tokens_in = _int(usage.get("prompt_tokens", usage.get("input_tokens")))
    tokens_out = _int(usage.get("completion_tokens", usage.get("output_tokens")))
    image_in = _int((usage.get("input_tokens_details") or {}).get("image_tokens")) if "image_input" in price else 0
    total = (price.get("input", 0) * max(tokens_in - image_in, 0) + price.get("image_input", 0) * image_in
             + price.get("output", 0) * tokens_out + price.get("image", 0) * 1_000_000 * _int(usage.get("generated_images"))
             + price.get("request", 0) * 1_000_000 * _int(usage.get("requests"))
             + price.get("second", 0) * 1_000_000 * _float(usage.get("seconds")))
    return round(total)
