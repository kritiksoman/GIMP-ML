#!/usr/bin/env python3
"""Build the lightweight macOS disk image using Apple's standard tools."""

import argparse
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent.parent


def build_native_installer() -> tuple[Path, Path]:
    package = ROOT / "scripts/macos/installer"
    build = ["swift", "build", "--package-path", str(package), "-c", "release",
             "--arch", "arm64", "--arch", "x86_64"]
    subprocess.run([*build, "--product", "GIMPMLInstaller"], check=True)
    binaries = Path(subprocess.run([*build, "--show-bin-path"], check=True,
                                   capture_output=True, text=True).stdout.strip())
    return binaries / "GIMPMLInstaller", package / ".build/checkouts/SwiftTerm/LICENSE"


def stage_installer(directory: Path) -> Path:
    executable, license_file = build_native_installer()
    application = directory / "Install GIMP-ML.app"
    executables = application / "Contents/MacOS"
    executables.mkdir(parents=True)
    resources = application / "Contents/Resources"
    resources.mkdir()
    shutil.copy2(executable, executables / "GIMPMLInstaller")
    with (application / "Contents/Info.plist").open("wb") as info:
        plistlib.dump({
            "CFBundleIdentifier": "org.gimpml.installer",
            "CFBundleName": "Install GIMP-ML",
            "CFBundleExecutable": "GIMPMLInstaller",
            "CFBundlePackageType": "APPL",
            "CFBundleVersion": "1.3",
            "CFBundleShortVersionString": "1.3",
            "LSMinimumSystemVersion": "13.0",
            "NSHighResolutionCapable": True,
        }, info)
    shutil.copy2(license_file, resources / "SwiftTerm-LICENSE")
    shutil.copy2(ROOT / "scripts/macos/install.command", resources / "install.command")
    shutil.copy2(ROOT / "scripts/macos/install-session.command",
                 resources / "install-session.command")
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
        parser.error("Build the DMG on macOS with Swift 6.2+, Xcode, and hdiutil.")
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
