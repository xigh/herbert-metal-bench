"""The table of a Qwen speed campaign (bench/qwen-campaign.sh): pp512 and
tg128, mean ± standard deviation of the timed runs, read from each engine's
own output, untouched.

usage: python3 -I campaign_table.py CAMPAIGN_DIR
"""
import os, re, statistics, sys

root = sys.argv[1]
MODELS = ["Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.8-27B", "Qwen3.6-35B-A3B"]
ENGINES = ["herbert", "llama", "mlx", "rapid", "basert"]
NAMES = {"herbert": "herbert-metal-bf16", "llama": "llama.cpp", "mlx": "mlx-lm", "rapid": "Rapid-MLX", "basert": "BaseRT"}
cell = re.compile(r"([0-9.]+) ± ([0-9.]+)")


def parse(engine, text):
    out = {}
    if engine == "mlx":
        pp = [float(x) for x in re.findall(r"Trial \d+:\s+prompt_tps=([0-9.]+)", text)]
        tg = [float(x) for x in re.findall(r"generation_tps=([0-9.]+), peak", text)]
        for key, v in (("pp512", pp), ("tg128", tg)):
            if v:
                out[key] = (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0)
        return out
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        for key in ("pp512", "tg128"):
            if any(c == key or c.startswith(key + ":") for c in cols):
                m = cell.search(cols[-1])
                if m:
                    out[key] = (float(m.group(1)), float(m.group(2)))
    return out


rows = {}
for m in MODELS:
    for e in ENGINES:
        p = os.path.join(root, m, f"{e}.log")
        if os.path.exists(p):
            rows[(m, e)] = parse(e, open(p, errors="replace").read())

for key, title in (("pp512", "pp512 (tok/s)"), ("tg128", "tg128 (tok/s)")):
    print(f"\n{title}\n")
    print("| model | " + " | ".join(NAMES[e] for e in ENGINES) + " |")
    print("| --- |" + " ---: |" * len(ENGINES))
    for m in MODELS:
        vals = []
        best = max((rows.get((m, e), {}).get(key, (0, 0))[0] for e in ENGINES), default=0)
        for e in ENGINES:
            v = rows.get((m, e), {}).get(key)
            if v is None:
                vals.append("—")
            else:
                s = f"{v[0]:,.1f} ± {v[1]:.1f}".replace(",", " ")
                vals.append(f"**{s}**" if v[0] == best and best > 0 else s)
        print(f"| {m} | " + " | ".join(vals) + " |")
