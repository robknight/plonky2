"""Critical-path probe: delay one timing scope per process and record whole-proof medians.

Usage: critical_path.py BINARY OUT [--scopes FILE] [--delay MS] [--sleep S]
       [--proof-scope NAME] [--skip N] [--binary-arg ARG ...] [--verified-pattern REGEX]

BINARY is a caller-supplied proving process built with gpu-profiling. It must
print TimingTree logs with a root named --proof-scope (default: prove).
Use --binary-arg for its arguments; it should run several comparable proofs.

Each listed timing scope is delayed by --delay ms at every start (PLONKY2_DELAY_SCOPE /
PLONKY2_DELAY_MS, gpu-profiling builds), in fresh processes, twice per scope, with
undelayed processes interleaved every four scopes. Writes OUT/results.json with each
process's warm-proof median (--skip leading proofs excluded) and the scope's count
in the last proof. --verified-pattern optionally counts verification log messages;
without it verification is unreported, so the caller must check the proofs.
"""
import argparse, json, os, re, statistics, subprocess, time

SCOPES = [
    "gpu: FFT and salt", "gpu: pack and enqueue salt writes",
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
p.add_argument("--proof-scope", default="prove", help="root TimingTree scope for one proof")
p.add_argument("--skip", type=int, default=1, help="leading proofs to exclude per process")
p.add_argument("--binary-arg", action="append", default=[], help="argument for BINARY; use --binary-arg=--flag for flags")
p.add_argument("--verified-pattern", help="regex matching one log message per verified proof")
a = p.parse_args()
if a.skip < 0:
    p.error("--skip must be nonnegative")
proof_line = re.compile(r"^(?:.*\][ \t]*)?([0-9.]+)s to " + re.escape(a.proof_scope) + r"$", re.M)
verified_line = re.compile(a.verified_pattern) if a.verified_pattern else None
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
        rc = subprocess.run([a.binary, *a.binary_arg], env=env, stdout=f, stderr=subprocess.STDOUT, timeout=900).returncode
    text = open(log).read()
    matches = list(proof_line.finditer(text))
    proofs = [float(m.group(1)) * 1000 for m in matches]
    verified = len(list(verified_line.finditer(text))) if verified_line else None
    last = text[matches[-1].end():] if matches else ""
    count = len(re.findall(r"s to " + re.escape(scope) + r"$", last, re.M)) if scope else 0
    r = dict(n=n, scope=scope, rc=rc, verified=verified, median=statistics.median(proofs[a.skip:]) if len(proofs) > a.skip else None,
             proofs=proofs, per_proof=count, delay_ms=a.delay)
    results.append(r)
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    print(n, scope, rc, verified, r["median"], count, flush=True)
