# NVIDIA check

Packages this repository and runs its GPU library suites and recursive proof example on a disposable NVIDIA Linux machine. No sibling checkouts or application-specific harnesses are required.

- `bundle.sh REPO OUT` exports the repository's committed HEAD and copies its `Cargo.lock` into `OUT/bundle.tar.gz`, together with `job.sh`. It refuses tracked changes and prints the revision and bundle's SHA-256. Generate a lockfile with `cargo generate-lockfile` first if the checkout has none.
- `job.sh` installs the toolchain named in `rust-toolchain` and the Vulkan loader, then runs the library suites with `gpu` and `gpu-profiling` and the `bench_recursion` example. Each step logs to `results/`; a nonzero exit status reports a failed step.

## Procedure

1. Make the bundle from the repository root: `tools/gpu/nvidia-check/bundle.sh . /tmp/nvidia-check`.
2. Provision an NVIDIA Linux machine with its GPU driver, Ubuntu's `apt-get`, root access and enough disk space for a release build. Expose SSH for file transfer and execution.
3. Obtain the machine's SSH host key fingerprint from its provider or console. Pin the key in `known_hosts` and compare its fingerprint before connecting.
4. Upload the bundle to a working directory, verify the printed SHA-256 with `sha256sum -c`, extract it and run `./job.sh`. It runs relative to its own directory.
5. Watch `results/steps.txt` (`<step> <exit code> <seconds>` per step) until `results/DONE` exists.
6. Fetch `results/` and release the machine.

## Notes

- The NVIDIA Vulkan driver needs `libegl1`; `job.sh` installs it together with `libvulkan1` and `vulkan-tools`.
- Fetching crates can time out on some hosts; `job.sh` retries the fetch once and raises Cargo's retry count and HTTP timeout.
- The GPU suites include proof verification, compiled-witness comparisons, program serialization and shutdown tests. Hardware regression tests fail if a usable GPU is unavailable.
