#!/usr/bin/env python3 -I
"""The tg128 table of the article #004 complement: herbert bf16 (real time and GPU time), mlx-lm at -p 1 and -p 16,
llama.cpp bf16, from the logs of runs/art004/_art004-mlxtg.sh, next to the mlx-lm value already published (after
a 2 048-token prompt). usage: bench/mlxtg_table.py LOGS_DIR"""
import re
import sys
from pathlib import Path

MODELS = ["Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.8-27B", "Qwen3.6-35B-A3B"]
PUBLISHED = {"Qwen3.5-0.8B": "250", "Qwen3.5-2B": "144", "Qwen3.5-4B": "71", "Qwen3.5-9B": "41",
             "Qwen3.8-27B": "ne charge pas", "Qwen3.6-35B-A3B": "73 à 78"}


def read(p):
    return p.read_text(errors="replace") if p.exists() else ""


def herbert(text):
    """(real, gpu) tg128 cells of herbert-metal-bench's two tables."""
    real = gpu = None
    table = None
    for line in text.splitlines():
        if line.startswith("| model |"):
            table = "gpu" if "gpu t/s" in line else "real"
        elif line.startswith("|") and "| tg128 |" in line:
            cell = line.split("|")[-2].strip()
            if table == "gpu":
                gpu = cell
            else:
                real = cell
    return real, gpu


def mlx(text):
    m = re.search(r"Averages:.*generation_tps=([\d.]+)", text)
    if m:
        return f"{float(m.group(1)):.1f}"
    if "GPU Timeout" in text:
        return "GPU Timeout"
    if "Error" in text or "Traceback" in text:
        return "erreur"
    return "" if not text else "?"


def llama(text):
    for line in text.splitlines():
        if line.startswith("|") and "tg128" in line:
            return line.split("|")[-2].strip()
    return ""


def main():
    root = Path(sys.argv[1])
    print("| modèle | herbert bf16 (réel) | herbert bf16 (GPU) | mlx-lm -p 1 | mlx-lm -p 16 | llama.cpp bf16 | mlx-lm publié (pp2048) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for m in MODELS:
        d = root / m
        if not d.exists():
            continue
        real, gpu = herbert(read(d / "h-bf16.log"))
        p1 = mlx(read(d / "mlx-p1.log"))
        if (d / "mlx-p1-retry.log").exists():
            p1 = f"{p1}, relancé une fois : {mlx(read(d / 'mlx-p1-retry.log'))}"
        print(f"| {m} | {real or ''} | {gpu or ''} | {p1} | {mlx(read(d / 'mlx-p16.log'))} | {llama(read(d / 'llama-bf16.log'))} | {PUBLISHED[m]} |")


if __name__ == "__main__":
    main()
