#!/usr/bin/env bash
# Makes OUT/bundle.tar.gz from this repository's committed HEAD and Cargo.lock,
# then prints the revision, bundle size and SHA-256. Refuses tracked changes.
# Usage: bundle.sh REPO OUT
set -euo pipefail
REPO=$(git -C "$1" rev-parse --show-toplevel)
OUT=$2
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
    echo "repository has uncommitted changes to tracked files" >&2
    exit 1
fi
if [ ! -f "$REPO/Cargo.lock" ]; then
    echo "generate Cargo.lock with cargo generate-lockfile before bundling" >&2
    exit 1
fi
rm -rf "$OUT/src" "$OUT/bundle.tar.gz"
mkdir -p "$OUT/src"
echo "plonky2 $(git -C "$REPO" rev-parse HEAD)"
git -C "$REPO" archive --prefix=src/plonky2/ HEAD | tar -x -C "$OUT"
cp "$REPO/Cargo.lock" "$OUT/src/plonky2/"
cp "$OUT/src/plonky2/tools/gpu/nvidia-check/job.sh" "$OUT/job.sh"
(cd "$OUT" && COPYFILE_DISABLE=1 tar -czf bundle.tar.gz src job.sh)
ls -l "$OUT/bundle.tar.gz" | awk '{print "bytes", $5}'
shasum -a 256 "$OUT/bundle.tar.gz"
