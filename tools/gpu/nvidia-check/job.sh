#!/usr/bin/env bash
# Check on a disposable NVIDIA pod (see README.md): plonky2's GPU library
# suites, pod2's library suite and its MainPod tests with the witness
# executor (including the executor-against-worklist test), MainPod proofs on
# the worklist and with two workers, and mainpod::prepare with the disk cache,
# all with the default GpuConfig. Runs from the bundle bundle.sh makes. Each
# step writes results/<step>.log and appends "<step> <exit code> <seconds>" to
# results/steps.txt; results/DONE marks the end.
set -u
BASE=/workspace/job
R=$BASE/results
BIN=$BASE/bin
mkdir -p "$R" "$BIN"
export PATH=/root/.cargo/bin:$PATH
export CARGO_TARGET_DIR=$BASE/target
export DEBIAN_FRONTEND=noninteractive
export CARGO_NET_RETRY=10 CARGO_HTTP_TIMEOUT=120
TC=+nightly-2025-07-02
SRC=$BASE/src
POD2=$SRC/pod2-gpu-integrated
CACHE=${XDG_CACHE_HOME:-$HOME/.cache}/pod2

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
    apt-get update -qq
    apt-get install -y -qq build-essential pkg-config libvulkan1 vulkan-tools libegl1 libssl-dev curl
    if [ ! -x /root/.cargo/bin/rustup ]; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh
        sh /tmp/rustup-init.sh -y --profile minimal --default-toolchain nightly-2025-07-02
    fi
    rustup toolchain install nightly-2025-07-02 --profile minimal
}

hardware() {
    nvidia-smi
    vulkaninfo --summary
    nproc
    free -g
}

fetch() {
    cd "$POD2" && cargo $TC fetch --locked
    cd "$SRC/plonky2-integrated" && cargo $TC fetch
}

plonky2_gpu() {
    cd "$SRC/plonky2-integrated/plonky2" && cargo $TC test --release --features gpu --lib
}

plonky2_gpu_profiling() {
    cd "$SRC/plonky2-integrated/plonky2" && cargo $TC test --release --features gpu-profiling --lib
}

# Each binary is copied out, since later builds of the same name overwrite it.
build_bin() {
    local out=$1 bin=$2
    shift 2
    cd "$POD2" && cargo $TC build --release --locked "$@" --bin "$bin" && cp "$CARGO_TARGET_DIR/release/$bin" "$BIN/$out"
}

worklist_build() {
    build_bin prove_worklist mainpod_prove --features gpu
}

mainpod_worklist() {
    RUST_LOG=plonky2::gpu=info "$BIN/prove_worklist" --proofs 4
}

mainpod_workers() {
    RUST_LOG=plonky2::gpu=info "$BIN/prove_worklist" --proofs 6 --workers 2
}

prepare_build() {
    build_bin prove_prepare mainpod_prove --no-default-features \
        --features backend_plonky2,zk,disk_cache,gpu,witness_executor
}

# First process from an empty cache compiles and stores; the second loads.
prepare_cold() {
    rm -rf "$CACHE"
    RUST_LOG=plonky2::gpu=info,pod2::backends::plonky2::witness_program=debug "$BIN/prove_prepare" --prepare --proofs 3
}

prepare_warm() {
    RUST_LOG=plonky2::gpu=info,pod2::backends::plonky2::witness_program=debug "$BIN/prove_prepare" --prepare --proofs 3
}

pod2_gpu() {
    cd "$POD2" && cargo $TC test --release --locked --features gpu --lib
}

pod2_executor_tests() {
    cd "$POD2" && cargo $TC test --release --locked --features gpu,witness_executor --lib mainpod
}

step setup setup || { touch "$R/DONE"; exit 1; }
step hardware hardware
step fetch fetch || step fetch_retry fetch
step plonky2_gpu plonky2_gpu
step worklist_build worklist_build && {
    step mainpod_worklist mainpod_worklist
    step mainpod_workers mainpod_workers
}
step prepare_build prepare_build && {
    step prepare_cold prepare_cold
    step prepare_warm prepare_warm
}
step pod2_gpu pod2_gpu
step pod2_executor_tests pod2_executor_tests
step plonky2_gpu_profiling plonky2_gpu_profiling
touch "$R/DONE"
