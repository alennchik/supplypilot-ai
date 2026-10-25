#!/usr/bin/env bash
# Runs a command and, if it fails, repeats the end of its output as error
# annotations, so the cause shows on the pull request without opening the log.
set -uo pipefail

log="$(mktemp)"
"$@" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}

if [ "$status" -ne 0 ]; then
  tail -n 10 "$log" | while IFS= read -r line; do
    line="${line//'%'/'%25'}"
    line="${line//$'\r'/}"
    if [ -n "${line//[[:space:]]/}" ]; then
      echo "::error title=Command failed::${line}"
    fi
  done
fi

rm -f "$log"
exit "$status"
