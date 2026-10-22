"""Summarise critical_path.py results: each scope's share of its injected delay that reached the proof."""
import json, statistics, sys, collections
r = json.load(open(sys.argv[1]))
base = [x["median"] for x in r if x["scope"] is None and x["median"] is not None and x["rc"] == 0]
if not base:
    sys.exit("no successful undelayed proof timings found")
b = statistics.median(base)
print(f"undelayed process medians: {', '.join(f'{m:.0f}' for m in base)}; median {b:.1f} ms")
by = collections.defaultdict(list)
for x in r:
    if x["scope"]:
        by[x["scope"]].append(x)
print(f"{'scope':52} {'n':>2} {'delay':>6} {'proof +ms':>12} {'share':>6}")
for s, xs in by.items():
    occ = int(max(x["per_proof"] for x in xs))
    delay = xs[0].get("delay_ms", 25.0)
    d = [x["median"] - b for x in xs if x["median"] is not None]
    if not d:
        print(f"{s[:52]:52} CHECK: no proof timings")
        continue
    frac = statistics.mean(d) / (delay * occ) if occ else float("nan")
    ok = all(x["rc"] == 0 and x["median"] is not None and
             (x["verified"] is None or x["verified"] == len(x["proofs"])) for x in xs)
    print(f"{s[:52]:52} {occ:2d} {delay * occ:6.0f} {', '.join(f'{v:+.0f}' for v in d):>12} {frac:6.0%}{'' if ok else ' CHECK'}")
reported = [x["verified"] for x in r if x["verified"] is not None]
print("verified proofs:", sum(reported) if reported else "unreported")
