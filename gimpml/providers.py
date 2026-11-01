"""Provider adapters returning validated PNGs, without logging remote bodies."""

import base64
import binascii
import io
import json
import warnings
from dataclasses import dataclass

import requests
from PIL import Image, UnidentifiedImageError

from gimpml.settings import (
    MAX_REFERENCE_BYTES, GenerationError, GenerationRequest, api_key,
)


MAX_RESPONSE_BYTES = 48 * 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 32_000_000


@dataclass(frozen=True)
class GeneratedImage:
    png: bytes
    width: int
    height: int


def post_json(url: str, headers: dict[str, str], payload: dict, timeout: float) -> dict:
    try:
        with requests.post(url, headers=headers, json=payload, timeout=(15, timeout),
                           stream=True, allow_redirects=False) as response:
            if response.status_code != 200:
                messages = {
                    400: "The provider rejected the request. Check model availability and image settings.",
                    401: "The provider rejected the API key.",
                    402: "Insufficient provider credits.",
                    403: "The provider denied access. Check key permissions, region and model access.",
                    404: "This image model is unavailable on the selected provider.",
                    429: "Provider rate limit reached. Wait before generating again.",
                }
                raise GenerationError(messages.get(
                    response.status_code,
                    f"Provider returned HTTP {response.status_code}. Try again later; no automatic retry was made."
                ))
            body = bytearray()
            for chunk in response.iter_content(65536):
                body.extend(chunk)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise GenerationError("The provider response exceeds the supported size limit.")
    except requests.Timeout as error:
        raise GenerationError("Generation timed out. The provider may still charge for this request.") from error
    except requests.RequestException as error:
        raise GenerationError("Cannot connect to the provider. Check your internet connection.") from error
    try:
        result = json.loads(body)
    except ValueError as error:
        raise GenerationError("The provider returned invalid JSON.") from error
    if not isinstance(result, dict):
        raise GenerationError("The provider returned an unexpected response.")
    return result


def decode_image(encoded: str, mime_type: str) -> GeneratedImage:
    if mime_type not in ("image/png", "image/jpeg", "image/webp"):
        raise GenerationError("The provider returned an unsupported image format.")
    if not isinstance(encoded, str) or len(encoded) > MAX_IMAGE_BYTES * 4 // 3 + 4:
        raise GenerationError("The generated image exceeds the supported size limit.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise GenerationError("The provider returned invalid image data.") from error
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if image.width * image.height > MAX_PIXELS:
                    raise GenerationError("The generated image exceeds 32 million pixels.")
                image.load()
                output = io.BytesIO()
                image.convert("RGBA").save(output, format="PNG")
                return GeneratedImage(output.getvalue(), image.width, image.height)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise GenerationError("The provider returned a corrupt or oversized image.") from error


def reference_data(png: bytes) -> str:
    if not isinstance(png, bytes) or not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise GenerationError("The editing input must be a PNG image.")
    if len(png) > MAX_REFERENCE_BYTES:
        raise GenerationError("The editing input exceeds 12 MiB. Reduce the image size.")
    try:
        image = decode_image(base64.b64encode(png).decode(), "image/png")
    except GenerationError as error:
        raise GenerationError("Cannot use the editing input: " + str(error)) from error
    if len(image.png) > MAX_REFERENCE_BYTES:
        raise GenerationError("The editing input exceeds 12 MiB. Reduce the image size.")
    return base64.b64encode(image.png).decode()


def generate(request: GenerationRequest, timeout: float = 180,
             reference_png: bytes | None = None) -> GeneratedImage:
    request.validate()
    reference = reference_data(reference_png) if reference_png is not None else None
    key = api_key(request.provider)
    if request.provider == "openrouter":
        payload = {"model": f"google/{request.model}", "prompt": request.prompt,
                   "aspect_ratio": request.aspect_ratio, "n": 1}
        if request.model != "gemini-2.5-flash-image":
            payload["resolution"] = request.resolution
        if reference is not None:
            payload["input_references"] = [{
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + reference},
            }]
        result = post_json(
            "https://openrouter.ai/api/v1/images",
            {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            payload,
            timeout,
        )
        data = result.get("data")
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            raise GenerationError("OpenRouter returned no image. The prompt may have been blocked.")
        image = data[0]
        if not isinstance(image.get("b64_json"), str):
            raise GenerationError("OpenRouter returned no image data.")
        return decode_image(image["b64_json"], image.get("media_type", "image/png"))

    image_config = {"aspectRatio": request.aspect_ratio}
    if request.model != "gemini-2.5-flash-image":
        image_config["imageSize"] = request.resolution
    parts = [{"text": request.prompt}]
    if reference is not None:
        parts.append({"inlineData": {"mimeType": "image/png", "data": reference}})
    result = post_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{request.model}:generateContent",
        {"x-goog-api-key": key, "Content-Type": "application/json"},
        {"contents": [{"parts": parts}],
         "generationConfig": {"responseModalities": ["TEXT", "IMAGE"],
                              "imageConfig": image_config}},
        timeout,
    )
    feedback = result.get("promptFeedback", {})
    if isinstance(feedback, dict) and feedback.get("blockReason"):
        raise GenerationError("Google blocked this prompt. Revise it and try again.")
    candidates = result.get("candidates")
    if not isinstance(candidates, list):
        raise GenerationError("Google returned no image. The prompt may have been blocked.")
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content", {})
        parts = content.get("parts", []) if isinstance(content, dict) else []
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict) or part.get("thought"):
                continue
            inline = part.get("inlineData")
            if isinstance(inline, dict) and isinstance(inline.get("data"), str):
                return decode_image(inline["data"], inline.get("mimeType", ""))
    raise GenerationError("Google returned no generated image. The prompt may have been blocked.")
