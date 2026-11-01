"""One request on stdin, one JSON result on stdout. No persistent network service."""

import json
import sys
from pathlib import Path

from gimpml.providers import generate
from gimpml.settings import GenerationError, GenerationRequest, preferences


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read(65537))
        if not isinstance(payload, dict) or set(payload) != {"request", "output"}:
            raise GenerationError("Invalid worker request.")
        output = Path(payload["output"])
        if not output.is_absolute() or output.suffix != ".png" or not output.parent.is_dir():
            raise GenerationError("The worker requires an absolute PNG output path.")
        if output.exists():
            raise GenerationError("The output file already exists. Choose an unused output path.")
        request = GenerationRequest(**payload["request"])
        image = generate(request, timeout=preferences()["timeout_seconds"])
        with output.open("xb") as handle:
            handle.write(image.png)
        print(json.dumps({"status": "success", "width": image.width, "height": image.height}))
        return 0
    except GenerationError as error:
        print(json.dumps({"status": "error", "message": str(error)}))
        return 1
    except (OSError, ValueError, TypeError) as error:
        # Never send request data, filesystem contents, or SDK exception text to the UI.
        print(json.dumps({"status": "error",
                          "message": f"Invalid worker input or output ({type(error).__name__}). Check the installation."}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
