"""Critical-path probe: delay one timing scope per process and record whole-proof medians.

Usage: critical_path.py BINARY OUT [--scopes FILE] [--delay MS] [--sleep S] [--repeat N]

BINARY is pod2's mainpod_prove; it is run with --proofs N.

Each listed timing scope is delayed by --delay ms at every start (PLONKY2_DELAY_SCOPE /
PLONKY2_DELAY_MS, gpu-profiling builds; pod2 built with disk_cache for the compiled
witness executor), in fresh processes, twice per scope, with
undelayed processes interleaved every four scopes. Writes OUT/results.json with each
process's warm-proof median (first proof excluded) and the scope's count in the last proof.
"""
import argparse, json, os, re, statistics, subprocess, time

SCOPES = [
    "run 1587515 generators", "gpu: FFT and salt", "gpu: pack and enqueue salt writes",
    "gpu: submit leaf hashing with row layout, and levels", "gpu: hash levels and read cap",
    "gpu: read coefficients", "gpu: decode mapped readback", "gpu: partial products and Z's",
    "compute quotient polys", "split up quotient polys", "gpu: upload coefficients",
    "construct the opening set, including lookups", "gpu: initial polynomial and codeword readback",
    "fold codewords in the commitment phase", "find proof-of-work witness", "query rounds",
    "release prover buffers",
]
p = argparse.ArgumentParser()
p.add_argument("binary")
p.add_argument("out")
p.add_argument("--scopes", help="file with one timing scope name per line")
p.add_argument("--delay", type=float, default=25.0)
p.add_argument("--sleep", type=float, default=10.0, help="idle seconds before each process")
p.add_argument("--repeat", type=int, default=6, help="verified MainPod proofs per process")
a = p.parse_args()
scopes = [l.strip() for l in open(a.scopes) if l.strip()] if a.scopes else SCOPES
os.makedirs(a.out, exist_ok=True)
# Keep the caller's pool and thread settings; drop only earlier diagnostic switches.
base = {k: v for k, v in os.environ.items() if not k.startswith(("PLONKY2_DELAY_", "RUST_LOG"))}
base.update(RUST_LOG="plonky2::util::timing=debug")
plan = []
for rep in range(2):
    for i, s in enumerate(scopes):
        if i % 4 == 0:
            plan.append(None)
        plan.append(s)
plan.append(None)
results = []
for n, scope in enumerate(plan):
    time.sleep(a.sleep)
    env = dict(base)
    if scope:
        env.update(PLONKY2_DELAY_SCOPE=scope, PLONKY2_DELAY_MS=str(a.delay))
    log = os.path.join(a.out, f"{n:02d}.log")
    with open(log, "w") as f:
        rc = subprocess.run([a.binary, "--proofs", str(a.repeat)], env=env, stdout=f, stderr=subprocess.STDOUT, timeout=900).returncode
    text = open(log).read()
    proofs = [float(x) * 1000 for x in re.findall(r"\] ([0-9.]+)s to prove_rec_circuit", text)]
    verified = len(re.findall(r"verified MainPod proof run \d+ in", text))
    last = text.split("to prove_rec_circuit")[-2] if text.count("to prove_rec_circuit") >= 2 else text
    last = last.split("to prove_rec_circuit")[-1]
    count = len(re.findall(r"s to " + re.escape(scope) + r"$", last, re.M)) if scope else 0
    r = dict(n=n, scope=scope, rc=rc, verified=verified, median=statistics.median(proofs[1:]) if len(proofs) > 1 else None,
             proofs=proofs, per_proof=count, delay_ms=a.delay)
    results.append(r)
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    print(n, scope, rc, verified, r["median"], count, flush=True)
