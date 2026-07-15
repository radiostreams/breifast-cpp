#!/usr/bin/env bash
# Downloads the regression-test fixtures listed in tests/fixtures/manifest.txt from a public
# Nextcloud folder share into tests/fixtures/, skipping files that already exist locally. No
# credentials needed -- it's a public share; only scripts/upload_fixtures.sh (via rclone) needs
# write access.
#
# Usage:
#   ./scripts/fetch_fixtures.sh [--force]
#
# NEXTCLOUD_SHARE_URL can be overridden if the share link ever changes; defaults to the
# tests/fixtures/ public folder share.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

NEXTCLOUD_SHARE_URL="${NEXTCLOUD_SHARE_URL:-https://cld.chouh.fr/s/dkns3drnsmCQEPM}"

FORCE=0
if [ "${1:-}" = "--force" ]; then
  FORCE=1
fi

MANIFEST=tests/fixtures/manifest.txt
if [ ! -f "$MANIFEST" ]; then
  echo "error: $MANIFEST not found" >&2
  exit 1
fi

while IFS= read -r fixture; do
  [ -z "$fixture" ] && continue
  dest="tests/fixtures/$fixture"
  if [ -f "$dest" ] && [ "$FORCE" -ne 1 ]; then
    echo "skip (already present): $fixture"
    continue
  fi
  echo "fetching: $fixture"
  curl -fSL -G "${NEXTCLOUD_SHARE_URL}/download" --data-urlencode "path=/" --data-urlencode "files=${fixture}" \
    -o "$dest"
done < "$MANIFEST"

echo "done"
