#!/bin/bash
set -euo pipefail

resources="$(cd "$(dirname "$0")" && pwd)"
gimp_app="${1:?Supply the GIMP application path}"
gimp="$gimp_app/Contents/MacOS/gimp-console"
if [[ ! -x "$gimp" ]]; then
    gimp="$gimp_app/Contents/MacOS/gimp"
fi
if [[ ! -x "$gimp" ]]; then
    echo "This application does not contain a GIMP executable." >&2
    exit 1
fi

candidates=(/opt/homebrew/bin/python3 /usr/local/bin/python3
    /Library/Frameworks/Python.framework/Versions/Current/bin/python3 /usr/bin/python3)
for python in "${candidates[@]}"; do
    if [[ -x "$python" ]] && "$python" -c \
        'import sys, venv; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
        exec "$python" "$resources/payload/scripts/install.py" \
            --gimp "$gimp" \
            --runtime-dir "$HOME/Library/Application Support/GIMP-ML" \
            --register-plugin-path
    fi
done

echo "Python 3.10+ with venv is required. Install Python from python.org or Homebrew, then retry." >&2
exit 1
