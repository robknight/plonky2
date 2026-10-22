"""Per-proof GPU timeline from a Metal System Trace (xctrace) of a proving process.

Usage: timeline.py TRACE.trace PROCESS_PREFIX [--skip N] [--start-label REGEX] [--sequence]
       timeline.py --log RUN.log [--skip N] [--sequence]

Exports the GPU interval and encoder tables, joins each interval with its
compute pass label (the prover names every pass), splits the trace into proofs
at each pass matching --start-label (default: an inverse FFT after an idle
gap, which is the wires IFFT) together with the uploads just before it, keeps
the windows that contain a quotient pass, and reports per
proof the GPU span, busy time and idle time, the idle gaps by the passes on
either side, and GPU time by pass. --sequence prints the last proof's passes.
"""
import argparse, collections, re, statistics, subprocess, tempfile, os
import xml.etree.ElementTree as ET


def table(trace, schema):
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
        path = f.name
    xpath = f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'
    with open(path, "w") as f:
        subprocess.run(["xcrun", "xctrace", "export", "--input", trace, "--xpath", xpath], stdout=f, check=True)
    node = ET.parse(path).getroot().find("node")
    os.unlink(path)
    cols = [c.find("mnemonic").text for c in node.find("schema").findall("col")]
    ids, rows = {}, []

    def resolve(e):
        if "ref" in e.attrib:
            return ids[e.attrib["ref"]]
        if "id" in e.attrib:
            ids[e.attrib["id"]] = e
        for c in e:
            resolve(c)
        return e

    for row in node.findall("row"):
        rows.append({n: ((e.text or "").strip(), e.attrib.get("fmt", "")) for n, e in ((n, resolve(e)) for n, e in zip(cols, list(row)))})
    return rows


p = argparse.ArgumentParser()
p.add_argument("trace", nargs="?")
p.add_argument("process", nargs="?")
p.add_argument("--log", help="a process log with PLONKY2_GPU_PASS_TIMELINE=1 GPU_PASS lines (any platform); each flush is one proof")
p.add_argument("--skip", type=int, default=1, help="leading proofs to exclude (cold)")
p.add_argument("--start-label", default=r"^fft \d+ x 2\^\d+ inverse$")
p.add_argument("--sequence", action="store_true")
p.add_argument("--min-proof-ms", type=float, default=300, help="minimum time between proof starts")
a = p.parse_args()

if a.log:
    by_flush = collections.defaultdict(list)
    line = re.compile(r'GPU_PASS flush=(\d+) label="([^"]*)" begin_ns=(\d+) end_ns=(\d+)')
    for m in line.finditer(open(a.log).read()):
        by_flush[int(m.group(1))].append(dict(start=int(m.group(3)), end=int(m.group(4)), label=m.group(2)))
    proofs = [sorted(by_flush[k], key=lambda x: x["start"]) for k in sorted(by_flush)]
else:
    mine = lambda r: r["process"][1].startswith(a.process)
    enc = [r for r in table(a.trace, "metal-application-encoders-list") if mine(r)]
    labels = {r["encoder-id"][0]: r["encoder-label"][1] for r in enc}
    cb_labels = {r["cmdbuffer-id"][0]: r["cmdbuffer-label"][1] for r in enc}
    iv = []
    for r in table(a.trace, "metal-gpu-intervals"):
        if not mine(r):
            continue
        label = labels.get(r["encoder-id"][0], "")
        if label in ("", "Blit Command 0", "Compute Command 0"):
            label = f"[{cb_labels.get(r['cmdbuffer-id'][0], 'unlabelled')}]"
        iv.append(dict(start=int(r["start"][0]), end=int(r["start"][0]) + int(r["duration"][0]), label=label))
    iv.sort(key=lambda x: x["start"])

    start_re = re.compile(a.start_label)
    anchors = [i for i, x in enumerate(iv) if start_re.match(x["label"])]
    # A proof starts with the uploads just before its anchor (no gap above 25 ms).
    first = []
    for i in anchors:
        j = i
        while j > 0 and iv[j]["start"] - iv[j - 1]["end"] < 25e6:
            j -= 1
        # Later inverse FFTs of the same proof (Z, quotient) are not proof starts.
        if first and iv[j]["start"] - iv[first[-1]]["start"] < a.min_proof_ms * 1e6:
            continue
        if not first or j != first[-1]:
            first.append(j)
    proofs = [iv[first[k]:(first[k + 1] if k + 1 < len(first) else len(iv))] for k in range(len(first))]
    # Each window ends at the last pass before the gap that precedes the next proof.
    for k, w in enumerate(proofs):
        if k + 1 < len(proofs):
            while len(w) > 1 and w[-1]["start"] - max(x["end"] for x in w[:-1]) >= 25e6:
                w.pop()
# Keep proof windows that evaluated a quotient on the GPU.
proofs = [w for w in proofs if any(x["label"] == "quotient" for x in w)]
gaps, by_label, spans = collections.defaultdict(list), collections.defaultdict(float), []
for k, w in enumerate(proofs):
    busy, cur, prev_end, last = 0, None, None, None
    for x in w:
        if cur is None or x["start"] > cur[1]:
            if cur is not None:
                busy += cur[1] - cur[0]
                if k >= a.skip:
                    gaps[(last["label"], x["label"])].append((x["start"] - cur[1]) / 1e6)
            cur = [x["start"], x["end"]]
        else:
            cur[1] = max(cur[1], x["end"])
        if last is None or x["end"] >= last["end"]:
            last = x
        if k >= a.skip:
            by_label[x["label"]] += (x["end"] - x["start"]) / 1e6
    busy += cur[1] - cur[0]
    span = (max(x["end"] for x in w) - w[0]["start"]) / 1e6
    spans.append((span, busy / 1e6))

n = len(proofs) - a.skip
print("proof  GPU span ms  busy ms  idle ms")
for k, (span, busy) in enumerate(spans):
    print(f"{k:5d} {span:12.1f} {busy:8.1f} {span - busy:8.1f}{'  (excluded)' if k < a.skip else ''}")
print(f"\nIdle gaps per warm proof (n={n}), by the passes either side:")
total = 0
for (x, y), v in sorted(gaps.items(), key=lambda kv: -sum(kv[1])):
    t = sum(v) / n
    total += t
    if t >= 0.25:
        print(f"{t:7.2f} ms  x{len(v) / n:<4.0f} {x}  ->  {y}")
print(f"total {total:.1f} ms")
print(f"\nGPU time per warm proof by pass:")
for l, t in sorted(by_label.items(), key=lambda kv: -kv[1]):
    if t / n >= 0.25:
        print(f"{t / n:7.1f} ms  {l}")
if a.sequence:
    w, t0, prev = proofs[-1], proofs[-1][0]["start"], None
    print("\nLast proof, passes merged when adjacent:")
    merged = []
    for x in w:
        if merged and merged[-1]["label"] == x["label"] and x["start"] - merged[-1]["end"] < 0.2e6:
            merged[-1]["end"] = max(merged[-1]["end"], x["end"])
        else:
            merged.append(dict(x))
    for x in merged:
        gap = (x["start"] - prev) / 1e6 if prev is not None and x["start"] > prev else 0
        print(f"{(x['start'] - t0) / 1e6:8.1f} ms  {(x['end'] - x['start']) / 1e6:7.2f} ms  gap {gap:6.2f}  {x['label']}")
        prev = max(prev or 0, x["end"])
