"""Non-secret preferences and explicitly located API credentials."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ("gemini-2.5-flash-image", "gemini-3.1-flash-image", "gemini-3-pro-image")
RATIOS = ("1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "4:5", "5:4", "21:9")
RESOLUTIONS = ("1K", "2K", "4K")
PROVIDERS = ("openrouter", "google")
KEY_HELP_URLS = {
    "openrouter": "https://www.apideck.com/blog/how-to-get-your-openrouter-api-key",
    "google": "https://www.merge.dev/blog/gemini-api-key",
}
MAX_REFERENCE_BYTES = 12 * 1024 * 1024
MAX_REFERENCE_PIXELS = 32_000_000


class GenerationError(Exception):
    """An actionable error safe to display without credentials or response bodies."""


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    provider: str = "openrouter"
    model: str = MODELS[0]
    aspect_ratio: str = "1:1"
    resolution: str = "1K"

    def validate(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise GenerationError("Enter an image-generation prompt.")
        if len(self.prompt) > 16000:
            raise GenerationError("The prompt must contain at most 16,000 characters.")
        if self.provider not in PROVIDERS:
            raise GenerationError("Choose OpenRouter or Google.")
        if self.model not in MODELS:
            raise GenerationError("Choose a supported Gemini image-generation model.")
        if self.aspect_ratio not in RATIOS or self.resolution not in RESOLUTIONS:
            raise GenerationError("Choose a supported aspect ratio and resolution.")
        if self.model == MODELS[0] and self.resolution != "1K":
            raise GenerationError("Gemini 2.5 Flash Image supports 1K only. Choose another model for 2K/4K.")


def preferences(root: Path = ROOT) -> dict:
    try:
        values = json.loads((root / "gimpml" / "config.json").read_text())
    except (OSError, ValueError) as error:
        raise GenerationError("Cannot read gimpml/config.json. Check the installation.") from error
    if not isinstance(values, dict):
        raise GenerationError("gimpml/config.json must contain a JSON object.")
    required = {"provider", "model", "aspect_ratio", "resolution", "timeout_seconds"}
    if set(values) != required:
        raise GenerationError("gimpml/config.json contains missing or unsupported settings.")
    GenerationRequest(prompt="validate settings", **{
        key: values[key] for key in required if key != "timeout_seconds"
    }).validate()
    timeout = values["timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 10 <= timeout <= 600:
        raise GenerationError("timeout_seconds must be between 10 and 600.")
    return values


def api_key(provider: str) -> str:
    from gimpml.credentials import read_key

    name = "OPENROUTER_API_KEY" if provider == "openrouter" else "GEMINI_API_KEY"
    key = os.environ.get(name)
    if provider == "google" and not key:
        key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        key = read_key(provider)
    if not key or not key.strip():
        provider_name = "OpenRouter" if provider == "openrouter" else "Google"
        raise GenerationError(
            f"No {provider_name} API key found. Create one: {KEY_HELP_URLS[provider]}\n"
            f"Save it in Filters > GIMP-ML > Settings on macOS, or set {name} "
            "in the worker environment."
        )
    return key.strip()
