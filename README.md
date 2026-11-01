# herbert-metal-bench

Reproducible benchmarks of [herbert-metal-bf16](https://github.com/xigh/herbert-metal-bf16) against mlx-lm and llama.cpp (and, in the earlier campaigns, Rapid-MLX and BaseRT) on Apple GPUs, Qwen3.5-0.8B to Qwen3.6-35B-A3B, one request at a time: the protocol, the prediction written before the long prompts were measured, the scripts as they ran, and the raw logs. Companion to article #004 (Metal) on [philippe-anel.fr](https://philippe-anel.fr).

The question behind it: herbert-metal-bf16 writes its compute kernels at load, as Metal Shading Language, without `.metal` files, MLX or Metal Performance Shaders, and picks among several forms of each kernel by timing them on the chip. How does that hold against the reference engines of the Mac, from 512 to 81 920 tokens of prompt, and does it still give the same tokens?

Author: Philippe Anel. License: MIT, see `LICENSE`.

## Layout

```
protocol.md       what is measured and how, the campaign's points in order, the two fixes of the night
predictions.md    the prediction written before the long prompts were measured (French, as written)
art004/           the drivers of the campaign of 2026-11-01 (_scripts/, runnable copies) and the table script
raw/              raw logs of each campaign, as produced, machine names and paths replaced
```

## Machines

The tables come from a Mac Studio M3 Ultra, 96 GB, macOS 26.5 (`m3ultra` in the logs), the only machine of the three where every model fits and the measure stays stable. A MacBook Air M1 (8 GB, `m1air`) and a MacBook Air M5 (32 GB, `m5air`) appear in the earlier campaigns.

## The campaign of 2026-11-01 (`raw/2026-11-01-metal-f16-art004`)

On the M3 Ultra only, the engine's defaults (`gemm.auto`, `sdpa.auto`, `dec.auto`: every kernel form, tile and block timed at load), nothing changed in the engine after the measures. Six points in order, the drivers in `art004/_scripts/` (`_art004-1.sh`, then `_art004-rest.sh`), every engine through `_f16-campaign.sh` (2 runs per cell, the order of the engines reversed from one cell to the next). The details are in `protocol.md`.

**The night the engine picked wrong kernels.** The campaign started at development commit `9e80538`. At 02:00 the token check (point 5) gave 0/48 identical ids for both the bf16 and the f16 engine: the `matmul2d` kernels on register fragments written for the M5 (`sdpa_nax`, `gemm_nax`, `gemm_nax_swiglu`) build on the M3 Ultra, are timed faster than the others there, so the load picked them, and return wrong values there (the mapping of matrix elements to lanes is the M5's). Isolated: `sdpa.prefill=4,gemm.mma=1` gave 48/48, `sdpa.prefill=3` alone or `gemm.mma=1` alone 0/48. Every herbert speed measured at `9e80538` was a speed of wrong kernels.

- **First fix, `2b86c37`** (from `1dcbd57`): each candidate of a load-time pick is run once against the first candidate, a kernel in production (`flash`, `mma`, `apart`), on a row count that is a multiple of no tile, the used rows compared, the K/V caches filled first (with zeros every kernel returned zeros and the check saw nothing); a difference above 1/64 of the largest magnitude rejects the candidate (`[autotune] … wrong against flash on 40 rows, never picked: nax`). Both engines gave 48/48 on the 0.8B's two prompts; two `gemm_nax` forms on 2048 × 1024 passed and stayed picked. herbert's measures were redone at this commit (`_art004-rerun.sh`).
- **Second fix, `f07e234`** (04:05, the published tree): point 5 redone at `2b86c37` still gave 0/48 on the long prompts of the 4B and the 9B: a `gemm_nax` form (64 × 64, band 2) passed the check on the output projection (4096 × 2560, 4096 × 4096) and was picked there. The check compared a partial row count with the dispatch groups of the full block, which made the comparison itself wrong (`mm2d` differed from `mma` there). Now each candidate runs on the full block and on a partial one with its dispatches regenerated for that row count, from a zeroed output: every nax form is rejected on the M3 Ultra, the 4B bf16 is back to its usual divergences at ties (35/48 and 28/48 against the bf16 reference), the 0.8B to 48/48. herbert's measures were redone a second time, at this commit.

llama.cpp's and mlx-lm's measures are those of the first pass of the same night (their engines did not change); herbert's are those at `f07e234`. The measures at `9e80538` (folders `1-bf16-long`, `3-d32768`, `4-big`, `5-ids`) and at `2b86c37` (`*-2b86c37`) are kept, marked as such.

The engines: herbert-metal-bf16 at `f07e234` (bf16, and f16 with `--features f16`), mlx-lm 0.32.0 on mlx 0.32.3 (bf16 checkpoint; the folder READMEs of the night say « mlx-lm 0.32.3 », the version of mlx), llama.cpp `fb4b2737` (Metal, GGUF BF16 and F16). mlx-lm's and llama.cpp's rates are their own benchmarks' (wall-clock). herbert's decode rates are wall-clock too (the first table of `herbert-metal-bench`); its GPU rate (the `gpu t/s` table, the GPU time of the command buffers, the one `art004/_art004_tables.py` reads) is 1 to 12 % higher there (Qwen3.5-0.8B tg128: 382 against 336) and is not used. herbert's prefill rates are its GPU rate, within 0.5 % of its wall-clock one (0.8B pp2048: 12 775 against 12 738).

### The same tokens (point 5, `logs/5r-ids`)

Two prompts per model, 48 greedy tokens. Identical ids before the first difference, short / long:

| model | herbert bf16 against mlx-lm bf16 | herbert f16 against mlx-lm float16 | herbert f16 against mlx-lm bf16 |
|---|---|---|---|
| Qwen3.5-0.8B | 48 / 48 | 48 / 48 | 48 / 43 |
| Qwen3.5-2B | 48 / 25 | 48 / 20 | 48 / 25 |
| Qwen3.5-4B | 48 / 28 | 48 / 48 | 48 / 48 |
| Qwen3.5-9B | 48 / 48 | 48 / 48 | 48 / 48 |
| Qwen3.8-27B | 15 / 13 | 48 / 48 | 20 / 13 |
| Qwen3.6-35B-A3B | 48 / 24 | 48 / 48 | 48 / 24 |

Every one of the 11 divergences is classed (`_tables-5r.md`): 9 exact ties of the reference (its two best tokens at the same logit, the summation order splits them) and 2 differences of one bf16 ulp (the reference separates its two best by 0.125, herbert puts them level or the other way: Qwen3.8-27B short, token 15, reference 27 / 26.875, herbert 27 / 27; Qwen3.5-0.8B long in f16 against bf16, token 43). No computation divergence.

### Prefill on long prompts, bf16 (point 1, `logs/1r-bf16-long`), tokens/s, herbert / mlx-lm / llama.cpp

| model | 2 048 | 8 192 | 24 576 | 61 440 |
|---|---|---|---|---|
| Qwen3.5-0.8B | 12 775 / 10 517 / 9 377 (×1.21) | 11 726 / 10 109 / 8 519 (×1.16) | 8 927 / 8 478 / 6 308 (×1.05) | 5 751 / 5 692 / 3 926 (×1.01) |
| Qwen3.5-2B | 5 884 / 5 541 / 4 723 (×1.06) | 5 676 / 5 433 / 4 492 (×1.04) | 4 932 / 4 829 / 3 784 (×1.02) | 3 771 / 3 816 / 2 773 (×0.99) |
| Qwen3.5-4B | 2 386 / 2 137 / 1 897 (×1.12) | 2 289 / 2 087 / 1 782 (×1.10) | 1 971 / 1 840 / 1 507 (×1.07) | 1 493 / 1 428 / 1 113 (×1.05) |
| Qwen3.5-9B | 1 315 / 1 209 / 1 115 (×1.09) | 1 286 / 1 200 / 1 075 (×1.07) | 1 178 / 1 118 / 969 (×1.05) | 989 / 954 / 790 (×1.04) |

The ratio is herbert over mlx-lm. Qwen3.5-0.8B at 81 920 tokens: 4 827 / 4 784 / 3 232 (×1.01). In the comparison of 2026-10-30 (`raw/2026-10-31-metal-f16-vs-engines`), before the prefill attention moved to the matrix units, the 0.8B made 1 128 tokens/s at 81 920 tokens against 4 848 for mlx-lm and 3 240 for llama.cpp. llama.cpp f16 runs within 1 % of its bf16 (point 2).

### The two large models (point 4, `logs/4r-big`), pp512 / pp8192 / tg128

| model | herbert f16 | herbert bf16 | llama.cpp bf16 | mlx-lm |
|---|---|---|---|---|
| Qwen3.8-27B | 374 / 365 / 14.8 | 374 / 365 / 14.7 | 320 / 307 / 13.5 | does not load (GPU Timeout, twice); loaded the next morning, tg128 13.4 from an empty context |
| Qwen3.6-35B-A3B | 2 430 / 3 267 / 87.1 | 2 415 / 3 248 / 87.0 | 2 004 / 1 804 / 71.7 | 1 893 / 2 487 / 73.3 to 78.4 (73.3 after pp512, 78.4 after pp8192; 74.0 from an empty context) |

### Decode, tokens/s

tg128, bf16, in the runs of point 1: herbert 336 / 170 / 82 / 46 (0.8B, 2B, 4B, 9B) and llama.cpp 210 / 135 / 69 / 41 from an empty context; mlx-lm 250 / 144 / 71 / 41 is `mlx_lm benchmark -p 2048 -g 128`, so its 128 tokens follow a 2 048-token prompt. mlx-lm measured again from an (almost) empty context the next morning, `-p 1` and `-p 16`, framed by herbert and llama.cpp in the same session, the order reversed from one model to the next (`raw/2026-11-01-metal-art004-mlx-tg-vide`):

| model | herbert bf16 | mlx-lm `-p 1` | mlx-lm `-p 16` | llama.cpp bf16 `-p 0` | herbert / mlx-lm |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 336.3 | 255.6 | 255.1 | 207.3 | ×1.32 |
| Qwen3.5-2B | 170.4 | 144.2 | 144.2 | 133.3 | ×1.18 |
| Qwen3.5-4B | 82.5 | 71.2 | 71.4 | 68.1 | ×1.16 |
| Qwen3.5-9B | 46.3 | 40.9 | 40.9 | 40.8 | ×1.13 |
| Qwen3.8-27B | 14.7 | 13.4 | 13.4 | 13.4 | ×1.10 |
| Qwen3.6-35B-A3B | 86.6 | 74.0 (reloaded once after a GPU Timeout) | 73.7 | 71.9 | ×1.17 |

The empty context does not change mlx-lm's decode: its values after 2 048 tokens hold within 2 %.

tg128 after 32 768 tokens (point 3, `logs/3r-d32768`) and after 81 920 tokens (point 3 bis, `logs/3-d81920`):

| context | model | herbert f16 | herbert bf16 | mlx-lm | llama.cpp bf16 | llama.cpp f16 |
|---|---|---:|---:|---:|---:|---:|
| 32 768 | Qwen3.5-0.8B | 269 | 268 | 201 | 174 | 177 |
| | Qwen3.5-2B | 146 | 146 | 125 | 121 | 122 |
| | Qwen3.5-4B | 70 | 67 | 62 | 59 | 60 |
| | Qwen3.5-9B | 41 | 42 | 38 | 38 | see `3-d32768/Qwen3.5-9B/llama-f16.log` |
| 81 920 | Qwen3.5-0.8B | 206 | 185 | 156 | 154 | 156 |
| | Qwen3.5-2B | 128 | 129 | 105 | 109 | 110 |
| | Qwen3.5-4B | 56 | 56 | 52 | 47 | 47 |
| | Qwen3.5-9B | 35 | 36.5 | 34 | 33 | 33 |

### The pieces (point 6, `logs/6r-pieces`)

- The attention's share of the 0.8B's prefill (f16, the kernels the defaults picked fixed with `--tune gemm.mma=2,gemm.tn=32,sdpa.prefill=3,sdpa.qreg=1`, one prefill counted, `profile-fixed-0.8B-pp*.log`): 6.1 % at 2 048 tokens (9.60 ms of 158.0 ms), 59.6 % at 61 440 (6 360 ms of 10 676 ms); with the scalar flash attention (`raw/2026-10-31-metal-f16-sdpa-long`) 21 % and 89.5 %. The default profile (6.4 % and 41.3 %) also counted the dispatches of the load-time measure.
- A load log (`dump-msl.log`): `[autotune] sdpa 512 rows at 512: flash 1.444 ms, mma 0.533 ms, mm2d 0.825 ms, nax* 0.461 ms -> mma` (the star: rejected by the check) and `[metal] 688 kernels compiled in 0.11 s, GEMMs and blocks timed in 24.11 s, 137 dispatches per token, blocks of 2048 rows`.
- The emitted MSL: `msl/` holds the 688 kernels emitted at load for the 0.8B in f16; `_sdpa_mma_h8kv2d256_q16k32_st_qr.metal` (the folder's root) is the prefill attention on the matrix units.

## Raw logs

`raw/` holds the raw logs, one folder per campaign, each with its own `README.md` written during the campaign (in French), the copies of the scripts it ran with (prefixed `_`), the tables and every output file.

| folder | measure |
|---|---|
| `2026-10-29-metal-bf16-qwen-vs-moteurs` | herbert-metal-bf16 (earlier tree `9e3c939`, bf16 only, the load-time measure of the GEMMs) against llama.cpp, mlx-lm, Rapid-MLX and BaseRT, six Qwen, pp512 and tg128, two rounds in reverse order: the first campaign where herbert led every prefill |
| `2026-10-31-metal-f16-vs-engines` | the f16 engine and its bf16 build against the same selection on the M3 Ultra and the M1 Air, pp128 to pp4096, tg128 after 4 096 tokens, then the long pass up to 81 920 tokens, where the prefill fell behind |
| `2026-10-31-metal-f16-sdpa-long` | why the prefill fell behind past ~2 000 tokens (the scalar attention's share, the prefill's methods forced one by one), then the attention on the matrix units (`sdpa_mma`, `sdpa_mm2d`, `sdpa.auto`), on the three machines; the probes `hacc` and `h2mv` |
| `2026-11-01-metal-f16-art004` | the article's campaign, above |
| `2026-11-01-metal-art004-mlx-tg-vide` | the next morning: mlx-lm's tg128 from an (almost) empty context on the six Qwen bf16, framed by herbert and llama.cpp in the same session |

The replacement is mechanical and the same everywhere (the scripts are in the development tooling, not here): the home directories become `~`, the external disks' model, GGUF and engine directories `$MODELS_DIR`, `$GGUF_DIR`, `$ENGINES_DIR` (with a default in the `.sh` copies, so that they still run), the machines `m3ultra`, `m1air` and `m5air` (in file names too), the user applications seen by the process listings `(user application)`, the local servers' address `localhost`, the development tree's directory `herbert-metal-bf16`, the llama.cpp build directory the commit it was built from (`llama.cpp-fb4b2737`); references to private working notes are removed or replaced by the folder that holds the measure, requests quoted in the headers are removed, and « jeton » reads « token » in the French prose. Rapid-MLX's start-up banner names a coding-agent integration; that name reads `<agent>`. No number is touched. The emitted kernels (`.metal`) only get the path rules.

## What this release leaves out

Published as a single commit without history. Left out of `raw/2026-10-31-metal-f16-vs-engines`: the M5 part (indicative, the machine heats and slows down, and it was used for other work during the campaign), the operator's journal, the HTML summary page and its script; the paragraphs of that folder's README that describe them are kept as written. The campaign of 2026-10-28 that `2026-10-29-metal-bf16-qwen-vs-moteurs` repeats is not in this release. Machine names, user names and local paths were replaced by neutral ones, as above. The measuring logic of the scripts is untouched.
