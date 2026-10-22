# NVIDIA check

Runs the GPU test suites and MainPod proofs on a disposable NVIDIA cloud machine, for changes that need checking on Vulkan and NVIDIA as well as on a Mac. On a Runpod A40 pod it takes about 25 minutes.

- `bundle.sh WORKSPACE OUT` exports the committed HEAD of plonky2-integrated, pod2-gpu-integrated and plonky2-u32-integrated (checked out side by side in `WORKSPACE`, since pod2's Cargo patches point at those sibling directories) with their lockfiles and `job.sh`, and prints the bundle's SHA-256.
- `job.sh` installs the toolchain and Vulkan loader, then runs each step with its log in `results/`: plonky2's library suites with `gpu` and `gpu-profiling`, MainPod proofs on the worklist (four proofs, then six with two workers), `mainpod_prove --prepare` from an empty disk cache and again with it filled, pod2's library suite with `gpu`, and pod2's MainPod tests with `gpu,witness_executor` (which include the executor-against-worklist and `prepare` tests).

The bundle contains unpublished source: upload it only where that is acceptable.

## Procedure

1. Make the bundle: `tools/gpu/nvidia-check/bundle.sh ~/path/to/workspace /tmp/nvidia-check`.
2. Create a pod with an NVIDIA GPU, a CUDA 12.8 or later image with Ubuntu 24.04 (we used `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`), a 60 GB container disk, a `/workspace` volume and SSH on TCP port 22.
3. Read the pod's boot log for its SSH host key fingerprints and pin the RSA key before connecting (`ssh-keyscan -p PORT -t rsa HOST > known_hosts`, then check `ssh-keygen -lf known_hosts` against the log).
4. Upload and start, checking the hash on the pod:

   ```sh
   O="-o UserKnownHostsFile=known_hosts -o StrictHostKeyChecking=yes"
   scp $O -P PORT /tmp/nvidia-check/bundle.tar.gz root@HOST:/workspace/
   ssh $O -p PORT root@HOST 'cd /workspace && echo "SHA256  bundle.tar.gz" | sha256sum -c &&
     mkdir -p job && tar --no-same-owner -xzf bundle.tar.gz -C job && cd job &&
     (nohup timeout 50m ./job.sh > job.out 2>&1 < /dev/null &)'
   ```

5. Watch `results/steps.txt` (`<step> <exit code> <seconds>` per step) until `results/DONE` exists.
6. Fetch only the logs (`tar -czf results.tgz results job.out` on the pod, then `scp`), and delete the pod.

## Notes

- The image lacks `libegl1`, which NVIDIA's Vulkan driver needs; without it `vulkaninfo` fails with `ERROR_INCOMPATIBLE_DRIVER`. `job.sh` installs it.
- Fetching crates can time out on some hosts; `job.sh` retries the fetch once and raises Cargo's retry count and HTTP timeout.
- `prepare_cold` clears pod2's disk cache (`~/.cache/pod2`) first, so it compiles and stores the witness programs; `prepare_warm` then loads them.
- Each binary is copied to `bin/` after its build, since later builds of the same binary with other features overwrite it.
