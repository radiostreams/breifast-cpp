#!/usr/bin/env bash
# Uploads any tests/fixtures/*.npz not yet listed in tests/fixtures/manifest.txt to the
# Nextcloud folder backing the public share fetch_fixtures.sh downloads from, then adds them to
# the manifest. Requires an rclone remote named "nextcloud" already configured for Nextcloud's
# WebDAV endpoint:
#   rclone config create nextcloud webdav url=https://cld.chouh.fr/remote.php/dav/files/breifast/ \
#     vendor=nextcloud user=breifast pass=<app-password>
# (this script doesn't handle that setup, only the sync -- the password lives obscured in
# ~/.config/rclone/rclone.conf, never in this file).
#
# Usage: ./scripts/upload_fixtures.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DEST=nextcloud:breifast-fixtures

MANIFEST=tests/fixtures/manifest.txt
touch "$MANIFEST"

shopt -s nullglob
for path in tests/fixtures/*.npz; do
  fixture="$(basename "$path")"
  if grep -qxF "$fixture" "$MANIFEST"; then
    echo "skip (already in manifest): $fixture"
    continue
  fi
  echo "uploading: $fixture"
  rclone copyto "$path" "${DEST}/${fixture}"
  echo "$fixture" >> "$MANIFEST"
done

echo "done -- don't forget to commit the updated manifest"
