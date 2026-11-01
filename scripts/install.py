#!/usr/bin/env python3
"""Install the plugin and its isolated backend without modifying GIMP.app."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parent.parent


def plugin_directory(executable: str | None) -> Path:
    executable = executable or (
        "/Applications/GIMP.app/Contents/MacOS/gimp-console" if sys.platform == "darwin"
        else shutil.which("gimp-console") or shutil.which("gimp")
    )
    if not executable:
        raise RuntimeError("Cannot find GIMP. Supply --gimp or --plug-in-dir.")
    result = subprocess.run([
        executable, "-n", "-d", "-f", "-s", "-c", "--batch-interpreter=python-fu-eval",
        "-b", "print('GIMPML_PLUGIN_DIR=' + Gimp.directory() + '/plug-ins')", "--quit",
    ], check=True, capture_output=True, text=True, timeout=120)
    for line in result.stdout.splitlines():
        if line.startswith("GIMPML_PLUGIN_DIR="):
            return Path(line.split("=", 1)[1])
    raise RuntimeError("Cannot discover GIMP's plugin directory. Supply --plug-in-dir.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plug-in-dir", type=Path)
    parser.add_argument("--gimp", help="GIMP executable used to discover its user plugin directory")
    args = parser.parse_args()
    directory = args.plug_in_dir or plugin_directory(args.gimp)
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(environment)
    subprocess.run([str(python), "-m", "pip", "install", "-r",
                    str(ROOT / "gimpml/requirements.txt")], check=True)
    subprocess.run([str(python), "-c",
                    "from gimpml.settings import preferences; preferences(); "
                    "import requests, PIL, dotenv; print('Backend ready')"],
                   cwd=ROOT, check=True)
    destination = directory.expanduser().resolve() / "gimpml-gemini"
    destination.mkdir(parents=True, exist_ok=True)
    source = ROOT / "plug-ins/gimpml-gemini/gimpml-gemini.py"
    target = destination / source.name
    shutil.copy2(source, target)
    target.chmod(0o755)
    (destination / "runtime.json").write_text(json.dumps(
        {"root": str(ROOT), "python": str(python)}, indent=2) + "\n")
    print(f"Installed: {target}\nRestart GIMP, then choose Filters > GIMP-ML > Generate Image with Gemini.")


if __name__ == "__main__":
    main()
