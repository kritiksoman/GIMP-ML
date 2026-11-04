#!/bin/bash
set -uo pipefail
umask 077

resources="${1:?Supply the resource directory}"
gimp_app="${2:?Supply the GIMP application path}"
log_file="${3:?Supply the log path}"
status_file="${4:?Supply the status path}"

publish_status() {
    result=$?
    trap - EXIT
    if ! printf '%s\n' "$result" > "$status_file.tmp" ||
        ! /bin/mv "$status_file.tmp" "$status_file"; then
        echo "Cannot publish installation exit status." >&2
        exit 1
    fi
    exit "$result"
}
trap publish_status EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

/bin/mkdir -p "$(dirname "$log_file")" &&
    /bin/chmod 700 "$(dirname "$log_file")" &&
    /usr/bin/touch "$log_file" &&
    /bin/chmod 600 "$log_file" || exit 1

export PYTHONUNBUFFERED=1
/bin/bash "$resources/install.command" "$gimp_app" 2>&1 | /usr/bin/tee "$log_file"
exit $?
