"""API credentials in macOS Keychain, accessed only by the external worker."""

import json
import sys

from gimpml.settings import GenerationError, PROVIDERS


SERVICE = "org.gimpml.api-keys"
MAX_KEY_LENGTH = 4096


def backend():
    if sys.platform != "darwin":
        raise GenerationError("Keychain settings are available on macOS only. Set API keys in the worker environment.")
    try:
        from keyring.backends.macOS import Keyring
    except ImportError as error:
        raise GenerationError("Keychain support is missing. Run scripts/install.py again.") from error
    return Keyring()


def read_key(provider: str) -> str | None:
    if provider not in PROVIDERS:
        raise GenerationError("Choose OpenRouter or Google.")
    if sys.platform != "darwin":
        return None
    store = backend()
    from keyring.errors import KeyringError

    try:
        return store.get_password(SERVICE, provider)
    except KeyringError as error:
        raise GenerationError("Cannot read macOS Keychain. Unlock it and allow access, then try again.") from error


def saved_status() -> dict[str, bool]:
    backend()
    return {provider: bool(read_key(provider)) for provider in PROVIDERS}


def save_keys(keys: dict[str, str], clear: list[str]) -> dict[str, bool]:
    if (not isinstance(keys, dict) or set(keys) - set(PROVIDERS)
            or not isinstance(clear, list) or any(provider not in PROVIDERS for provider in clear)
            or set(keys) & set(clear)):
        raise GenerationError("Invalid credential settings. Choose either a new key or removal for each provider.")
    for key in keys.values():
        if not isinstance(key, str) or not key.strip() or len(key) > MAX_KEY_LENGTH:
            raise GenerationError("API keys must contain 1 to 4,096 characters.")
    store = backend()
    from keyring.errors import KeyringError

    try:
        for provider, key in keys.items():
            store.set_password(SERVICE, provider, key.strip())
        for provider in clear:
            if store.get_password(SERVICE, provider) is not None:
                store.delete_password(SERVICE, provider)
    except KeyringError as error:
        raise GenerationError(
            "Cannot save macOS Keychain settings. Some changes may have been saved. "
            "Unlock Keychain and allow access, then reopen Settings to check."
        ) from error
    try:
        return saved_status()
    except GenerationError as error:
        raise GenerationError("Keychain changes were saved, but their status could not be read. "
                              "Unlock Keychain and reopen Settings to check.") from error


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read(20001))
        if not isinstance(payload, dict):
            raise GenerationError("Invalid credential request.")
        if payload == {"operation": "status"}:
            status = saved_status()
        elif set(payload) == {"operation", "keys", "clear"} and payload["operation"] == "save":
            status = save_keys(payload["keys"], payload["clear"])
        else:
            raise GenerationError("Invalid credential request.")
        print(json.dumps({"status": "success", "saved": status}))
        return 0
    except GenerationError as error:
        print(json.dumps({"status": "error", "message": str(error)}))
        return 1
    except (ValueError, TypeError):
        print(json.dumps({"status": "error", "message": "Invalid credential request."}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
