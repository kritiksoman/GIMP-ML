#!/usr/bin/env python3
"""Build the lightweight macOS disk image using Apple's standard tools."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent.parent


def stage_installer(directory: Path) -> Path:
    application = directory / "Install GIMP-ML.app"
    subprocess.run(["osacompile", "-o", str(application),
                    str(ROOT / "scripts/macos/installer.applescript")], check=True)
    resources = application / "Contents/Resources"
    shutil.copy2(ROOT / "scripts/macos/install.command", resources / "install.command")
    payload = resources / "payload"
    (payload / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/install.py", payload / "scripts/install.py")
    shutil.copytree(ROOT / "gimpml", payload / "gimpml",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    shutil.copytree(ROOT / "plug-ins", payload / "plug-ins",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "runtime.json"))
    for name in ("README.md", "LICENSE.md"):
        shutil.copy2(ROOT / name, directory / name)
    shutil.copytree(ROOT / "assets", directory / "assets",
                    ignore=shutil.ignore_patterns(".DS_Store"))
    subprocess.run(["codesign", "--force", "--sign", "-", str(application)], check=True)
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(application)], check=True)
    return application


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/GIMP-ML-macOS.dmg")
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("Build the DMG on macOS with osacompile and hdiutil.")
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"Output already exists: {output}. Choose another --output or remove it first.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gimpml-dmg-") as temporary:
        directory = Path(temporary)
        stage_installer(directory)
        subprocess.run(["hdiutil", "create", "-volname", "GIMP-ML Installer",
                        "-srcfolder", str(directory), "-format", "UDZO",
                        str(output)], check=True)
    print(f"Built: {output}")


if __name__ == "__main__":
    main()
