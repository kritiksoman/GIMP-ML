#!/usr/bin/env python3
"""Install the plugin and its isolated backend without modifying GIMP.app."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import venv


ROOT = Path(__file__).resolve().parent.parent


def gimp_configuration(executable: str | None) -> tuple[Path, str]:
    executable = executable or (
        "/Applications/GIMP.app/Contents/MacOS/gimp-console" if sys.platform == "darwin"
        else shutil.which("gimp-console") or shutil.which("gimp")
    )
    if not executable:
        raise RuntimeError("Cannot find GIMP. Supply --gimp or --plug-in-dir.")
    result = subprocess.run([
        executable, "-n", "-d", "-f", "-s", "-c", "--batch-interpreter=python-fu-eval",
        "-b", "import json; print('GIMPML_CONFIG=' + json.dumps("
        "[Gimp.directory(), Gimp.gimprc_query('plug-in-path')]))", "--quit",
    ], check=True, capture_output=True, text=True, timeout=120)
    for line in result.stdout.splitlines():
        if line.startswith("GIMPML_CONFIG="):
            directory, search_path = json.loads(line.split("=", 1)[1])
            if not isinstance(directory, str) or not isinstance(search_path, str):
                raise RuntimeError("GIMP returned an invalid plugin configuration.")
            return Path(directory), search_path
    raise RuntimeError("Cannot discover GIMP's plugin directory. Supply --plug-in-dir.")


def require_gimp_closed() -> None:
    result = subprocess.run(
        ["/usr/bin/pgrep", "-u", str(os.getuid()), "-x", r"gimp|gimp-console|gimp-3\.[0-9]+"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        raise RuntimeError("Fully quit GIMP (Cmd+Q), then run the installer again.")
    if result.returncode != 1:
        raise RuntimeError(f"Cannot check whether GIMP is running: {result.stderr.strip()}")


def register_plugin_path(config_directory: Path, directory: Path, search_path: str) -> None:
    if str(directory) in search_path.split(os.pathsep):
        return
    if os.pathsep in str(directory) or "\n" in str(directory) or "\r" in str(directory):
        raise RuntimeError("The plugin directory cannot contain a path separator or newline.")
    gimprc = config_directory / "gimprc"
    original = gimprc.read_text() if gimprc.exists() else ""
    pattern = re.compile(r'(?m)^[ \t]*\(plug-in-path\s+"((?:\\.|[^"\\])*)"\s*\)')
    matches = list(pattern.finditer(original))
    declarations = re.findall(r"(?m)^[ \t]*\(plug-in-path\b", original)
    if len(matches) != len(declarations) or len(matches) > 1:
        raise RuntimeError("Cannot safely update plug-in-path in gimprc. Configure it in GIMP Preferences.")
    escaped_directory = json.dumps(str(directory), ensure_ascii=False)[1:-1]
    if matches:
        match = matches[0]
        value = match.group(1) + os.pathsep + escaped_directory
        updated = original[:match.start()] + f'(plug-in-path "{value}")' + original[match.end():]
    else:
        value = json.dumps(search_path + os.pathsep + str(directory), ensure_ascii=False)
        updated = original.rstrip() + f"\n(plug-in-path {value})\n"
    config_directory.mkdir(parents=True, exist_ok=True)
    backup = gimprc.with_name("gimprc.gimpml-backup")
    if gimprc.exists() and not backup.exists():
        shutil.copy2(gimprc, backup)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=config_directory,
                                     prefix=".gimprc-gimpml-", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            temporary.write(updated)
            temporary.close()
            if gimprc.exists():
                shutil.copymode(gimprc, temporary_path)
            temporary_path.replace(gimprc)
        finally:
            temporary_path.unlink(missing_ok=True)
    print(f"Registered plugin search path in: {gimprc}")


def copy_runtime(destination: Path) -> None:
    if destination == ROOT:
        return
    if destination.is_relative_to(ROOT / "gimpml"):
        raise RuntimeError("The runtime directory must not be inside the source gimpml package.")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "gimpml", destination / "gimpml", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plug-in-dir", type=Path)
    parser.add_argument("--gimp", help="GIMP executable used to discover its user plugin directory")
    parser.add_argument("--runtime-dir", type=Path,
                        help="Copy the backend and create its venv at this permanent location")
    parser.add_argument("--register-plugin-path", action="store_true",
                        help="macOS: ensure GIMP searches the destination (GIMP must be closed)")
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.error("Python 3.10 or newer is required.")
    if args.register_plugin_path:
        if sys.platform != "darwin":
            parser.error("--register-plugin-path is currently supported only on macOS.")
        require_gimp_closed()
    config_directory, search_path = (None, "")
    if not args.plug_in_dir or args.register_plugin_path:
        config_directory, search_path = gimp_configuration(args.gimp)
    directory = args.plug_in_dir or config_directory / "plug-ins"
    directory = directory.expanduser().resolve()
    root = args.runtime_dir.expanduser().resolve() if args.runtime_dir else ROOT
    copy_runtime(root)
    environment = root / ".venv"
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(environment)
    subprocess.run([str(python), "-m", "pip", "install", "-r",
                    str(root / "gimpml/requirements.txt")], check=True)
    subprocess.run([str(python), "-c",
                    "from gimpml.settings import preferences; preferences(); "
                    "import requests, PIL; print('Backend ready')"],
                   cwd=root, check=True)
    if sys.platform == "darwin":
        subprocess.run([str(python), "-c", "from keyring.backends.macOS import Keyring; Keyring()"],
                       cwd=root, check=True)
    if args.register_plugin_path:
        require_gimp_closed()
        register_plugin_path(config_directory, directory, search_path)
    destination = directory / "gimpml-gemini"
    destination.mkdir(parents=True, exist_ok=True)
    source = ROOT / "plug-ins/gimpml-gemini/gimpml-gemini.py"
    target = destination / source.name
    shutil.copy2(source, target)
    target.chmod(0o755)
    (destination / "runtime.json").write_text(json.dumps(
        {"root": str(root), "python": str(python)}, indent=2) + "\n")
    print(f"Installed: {target}\nRestart GIMP, then choose Filters > GIMP-ML > "
          "Generate Image, Edit Selected Layer, Edit Visible Canvas, or Settings (macOS).")


if __name__ == "__main__":
    main()
