#!/usr/bin/env bash
# Makes OUT/bundle.tar.gz for job.sh: the committed HEAD of each repository
# (tracked files only, plus Cargo.lock) and job.sh, then prints its size and
# SHA-256. WORKSPACE holds the three checkouts side by side, under the names
# pod2's Cargo patches use: plonky2-integrated, pod2-gpu-integrated and
# plonky2-u32-integrated. Refuses a checkout with uncommitted changes to
# tracked files, so the bundle matches the recorded revisions.
#
# Usage: bundle.sh WORKSPACE OUT
set -euo pipefail
WORKSPACE=$(cd "$1" && pwd)
OUT=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
rm -rf "$OUT/src" "$OUT/bundle.tar.gz"
mkdir -p "$OUT/src"

# The tracked paths to export from each repository.
paths() {
    case $1 in
        plonky2-integrated) echo "Cargo.toml rust-toolchain rustfmt.toml .cargo field maybe_rayon plonky2 starky util" ;;
        pod2-gpu-integrated) echo "Cargo.toml build.rs rust-toolchain.toml rustfmt.toml src examples tests" ;;
        plonky2-u32-integrated) echo "Cargo.toml src" ;;
    esac
}

for repo in plonky2-integrated pod2-gpu-integrated plonky2-u32-integrated; do
    dir=$WORKSPACE/$repo
    if [ -n "$(git -C "$dir" status --porcelain --untracked-files=no)" ]; then
        echo "$repo has uncommitted changes to tracked files" >&2
        exit 1
    fi
    echo "$repo $(git -C "$dir" rev-parse --short HEAD) $(git -C "$dir" branch --show-current)"
    # shellcheck disable=SC2046,SC2086
    git -C "$dir" archive --prefix="$repo/" HEAD \
        $(git -C "$dir" ls-tree --name-only HEAD $(paths "$repo")) | tar -x -C "$OUT/src"
    if [ -f "$dir/Cargo.lock" ]; then
        cp "$dir/Cargo.lock" "$OUT/src/$repo/"
    fi
done

cp "$HERE/job.sh" "$OUT/job.sh"
(cd "$OUT" && COPYFILE_DISABLE=1 tar -czf bundle.tar.gz src job.sh)
ls -l "$OUT/bundle.tar.gz" | awk '{print "bytes", $5}'
shasum -a 256 "$OUT/bundle.tar.gz"
