#!/usr/bin/env python3
"""The tables of bench/_sdpa_long.sh's logs: per machine, the prefill speed (GPU tokens/s, the wall-clock one in
parentheses when it differs by more than 3 %) of every forced method at every prompt length, each as a ratio to the base
run of the same length; then, from the profile pass (HERBERT_METAL_PROFILE=1), the share of the attention kernel
(sdpa_*), the GEMMs (gemm_*), DeltaNet (dn_*) and the rest at every length, with the attention's time and the implied
rate in TFLOPS (QK and PV of the causal half, 2·2·hd flops per (query, key) pair, 6 full-attention layers of 8 heads
for the 0.8B: counted from the model's config when given). Markdown on stdout.

usage: bench/sdpa_long_table.py [--config CONFIG.json] <name>=<dir> ...   (e.g. m3ultra=runs/sdpa-long-m3ultra m1air=runs/sdpa-long-m1air)
"""
import json
import re
import sys
from pathlib import Path

RUN = re.compile(r"^\[pp(\d+)\] run ([\d.]+) s, ([\d.]+) tok/s, gpu ([\d.]+) s, ([\d.]+) tok/s")
PROF = re.compile(r"^\[profile prefill\]\s+([\d.]+) ms\s+([\d.]+) %\s+×(\d+)\s+([\d.]+) µs each\s+(\S+)")


def runs(log):
    """(pp, wall tok/s, gpu tok/s) of the timed runs (warm-ups excluded), or None."""
    rs = [RUN.match(l) for l in log.read_text(errors="replace").splitlines()]
    rs = [m for m in rs if m]
    if not rs:
        return None
    pp = int(rs[0].group(1))
    return pp, [float(m.group(3)) for m in rs], [float(m.group(5)) for m in rs]


def median(v):
    v = sorted(v)
    return v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2


def profile(log):
    """{kernel: seconds} and {kernel: dispatches} of the prefill's kernels."""
    d, n = {}, {}
    for l in log.read_text(errors="replace").splitlines():
        m = PROF.match(l)
        if m:
            d[m.group(5)] = d.get(m.group(5), 0.0) + float(m.group(1)) / 1e3
            n[m.group(5)] = n.get(m.group(5), 0) + int(m.group(3))
    return d, n


def machine(name, d, cfg):
    d = Path(d)
    out = [f"## {name} — {d}", ""]
    table = {}  # variant -> pp -> (wall, gpu)
    for log in sorted(d.glob("*-pp*.log")):
        m = re.match(r"(\w+)-pp(\d+)\.log$", log.name)
        if not m or m.group(1) == "profile":
            continue
        r = runs(log)
        if r is None:
            table.setdefault(m.group(1), {})[int(m.group(2))] = None
            continue
        table.setdefault(m.group(1), {})[r[0]] = (median(r[1]), median(r[2]), len(r[1]))
    pps = sorted({pp for v in table.values() for pp in v})
    variants = ["base"] + sorted(v for v in table if v != "base")
    out.append("| méthode | " + " | ".join(f"pp{pp}" for pp in pps) + " |")
    out.append("| --- | " + " | ".join("---:" for _ in pps) + " |")
    for v in variants:
        if v not in table:
            continue
        cells = []
        for pp in pps:
            c = table[v].get(pp)
            if c is None:
                cells.append("—" if pp in table[v] else "")
                continue
            wall, gpu, n = c
            s = f"{gpu:,.0f}".replace(",", " ")
            if abs(wall - gpu) / gpu > 0.03:
                s += f" ({wall:,.0f})".replace(",", " ")
            b = table.get("base", {}).get(pp)
            if v != "base" and b:
                s += f" ×{gpu / b[1]:.2f}"
            cells.append(s)
        out.append(f"| {v} | " + " | ".join(cells) + " |")
    out.append("")
    # The profile pass.
    profs, profs_counts = {}, {}
    for log in sorted(d.glob("profile-pp*.log")):
        pp = int(re.search(r"pp(\d+)", log.name).group(1))
        p, n = profile(log)
        if p:
            profs[pp], profs_counts[pp] = p, n
    if profs and cfg and 2048 in profs:
        # The load-time measurements (dec.auto's prefills of a block of 2047 ids) go through the same profile: at
        # pp2048 the attention kernel is dispatched `count` times for `layers` real ones, every extra dispatch a
        # block of 2048 from an empty context, the same as pp2048's own. That part of pp2048's profile, kernel by
        # kernel, is taken off every length.
        layers = sum(1 for t in cfg["layer_types"] if t == "full_attention")
        p0 = profs[2048]
        count = sum(v for k, v in profs_counts[2048].items() if k.startswith("sdpa_"))
        frac = (count - layers) / count if count > layers else 0.0
        out.append(f"Profil : {count} dispatchs d'attention à pp2048 pour {layers} couches, {count - layers} de la mesure au chargement ; {frac:.3f} du profil de pp2048 retiré, noyau par noyau, de chaque longueur.")
        out.append("")
        for pp in profs:
            profs[pp] = {k: v - frac * p0.get(k, 0.0) for k, v in profs[pp].items()}
    if profs:
        out.append("| profil (base) | " + " | ".join(f"pp{pp}" for pp in sorted(profs)) + " |")
        out.append("| --- | " + " | ".join("---:" for _ in profs) + " |")
        groups = [("attention (sdpa_*)", "sdpa_"), ("GEMM (gemm_*)", "gemm_"), ("DeltaNet (dn_*)", "dn_")]
        rows = {g: [] for g, _ in groups}
        rows["reste"] = []
        rows["GPU total (s)"] = []
        rows["attention (s)"] = []
        rows["attention, TFLOPS"] = []
        for pp in sorted(profs):
            p = profs[pp]
            total = sum(p.values())
            seen = 0.0
            for g, pre in groups:
                t = sum(v for k, v in p.items() if k.startswith(pre))
                seen += t
                rows[g].append(f"{100 * t / total:.1f} %")
            rows["reste"].append(f"{100 * (total - seen) / total:.1f} %")
            rows["GPU total (s)"].append(f"{total:.2f}")
            att = sum(v for k, v in p.items() if k.startswith("sdpa_"))
            rows["attention (s)"].append(f"{att:.3f}")
            if cfg:
                layers = sum(1 for t in cfg["layer_types"] if t == "full_attention")
                flops = layers * cfg["num_attention_heads"] * (pp * (pp + 1) / 2) * 4 * cfg["head_dim"]
                rows["attention, TFLOPS"].append(f"{flops / att / 1e12:.2f}" if att else "")
        for g, cells in rows.items():
            if any(cells):
                out.append(f"| {g} | " + " | ".join(cells) + " |")
        out.append("")
        # The attention kernels' own lines, by length.
        for pp in sorted(profs):
            p = profs[pp]
            total = sum(p.values())
            top = sorted(p.items(), key=lambda kv: -kv[1])[:4]
            out.append(f"pp{pp} : " + " ; ".join(f"{k} {100 * v / total:.1f} %" for k, v in top))
        out.append("")
    return out


def main(argv):
    cfg = None
    args = []
    it = iter(argv)
    for a in it:
        if a == "--config":
            cfg = json.load(open(next(it)))
            cfg = cfg.get("text_config", cfg)
        else:
            args.append(a)
    for a in args:
        name, d = a.split("=", 1)
        print("\n".join(machine(name, d, cfg)))


if __name__ == "__main__":
    main(sys.argv[1:])
