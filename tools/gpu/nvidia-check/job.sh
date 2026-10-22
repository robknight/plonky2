#!/usr/bin/env bash
# Runs this repository's GPU library suites and recursive proof example on a
# disposable NVIDIA machine. Run from the bundle produced by bundle.sh.
# Each step writes results/<step>.log and appends "<step> <exit code> <seconds>"
# to results/steps.txt; results/DONE marks the end.
set -u
BASE=$(cd "$(dirname "$0")" && pwd)
R=$BASE/results
SRC=$BASE/src/plonky2
mkdir -p "$R"
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=$BASE/target
export DEBIAN_FRONTEND=noninteractive
export CARGO_NET_RETRY=10 CARGO_HTTP_TIMEOUT=120
TC=$(tr -d '[:space:]' < "$SRC/rust-toolchain")

step() {
    local name=$1
    shift
    local started=$SECONDS
    ( "$@" ) > "$R/$name.log" 2>&1
    local code=$?
    echo "$name $code $((SECONDS - started))" >> "$R/steps.txt"
    return $code
}

setup() {
    apt-get update -qq &&
    apt-get install -y -qq build-essential pkg-config libvulkan1 vulkan-tools libegl1 libssl-dev curl || return
    if ! command -v rustup >/dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o "$BASE/rustup-init.sh" &&
        sh "$BASE/rustup-init.sh" -y --profile minimal --default-toolchain "$TC" || return
    fi
    rustup toolchain install "$TC" --profile minimal
}

hardware() {
    nvidia-smi && vulkaninfo --summary && nproc && free -g
}

fetch() {
    cd "$SRC" && cargo "+$TC" fetch --locked
}

plonky2_gpu() {
    cd "$SRC" && cargo "+$TC" test -p plonky2 --release --locked --features gpu --lib
}

plonky2_gpu_profiling() {
    cd "$SRC" && cargo "+$TC" test -p plonky2 --release --locked --features gpu-profiling --lib
}

recursive_proof() {
    cd "$SRC" && cargo "+$TC" run -p plonky2 --release --locked --features gpu \
        --example bench_recursion -- --size 14 --threads 1
}

step setup setup || { touch "$R/DONE"; exit 1; }
step hardware hardware || { touch "$R/DONE"; exit 1; }
step fetch fetch || step fetch_retry fetch || { touch "$R/DONE"; exit 1; }
failed=0
step plonky2_gpu plonky2_gpu || failed=1
step plonky2_gpu_profiling plonky2_gpu_profiling || failed=1
step recursive_proof recursive_proof || failed=1
touch "$R/DONE"
exit "$failed"
