"""Model pricing for cost tracking.

Prices are per 1 million tokens (input, output) in USD.
"""

MODEL_PRICES: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4.1":          (2.00, 8.00),
    "gpt-4.1-mini":     (0.40, 1.60),
    "gpt-4.1-nano":     (0.10, 0.40),
    "gpt-4o":           (2.50, 10.00),
    "gpt-4o-mini":      (0.15, 0.60),
    "gpt-3.5-turbo":    (0.50, 1.50),
    "o3":               (10.00, 40.00),
    "o3-mini":          (1.10, 4.40),
    "o4-mini":          (1.10, 4.40),
    # Anthropic
    "claude-opus-4-6":  (15.00, 75.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (0.80, 4.00),
    # Google
    "gemini-2.5-pro":   (1.25, 10.00),
    "gemini-2.5-flash": (0.15, 0.60),
    # Fallback
    "_default":         (1.00, 3.00),
}


def get_price(model: str) -> tuple[float, float]:
    """Return (input_per_1M, output_per_1M) for a model name."""
    if model in MODEL_PRICES:
        return MODEL_PRICES[model]
    lo = model.lower()
    for key, price in MODEL_PRICES.items():
        if key in lo:
            return price
    return MODEL_PRICES["_default"]


def calc_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in USD."""
    inp, out = get_price(model)
    return (prompt_tokens * inp + completion_tokens * out) / 1_000_000


def format_cost(usd: float) -> str:
    if usd < 0.01:
        return f"${usd:.4f}"
    return f"${usd:.3f}"
