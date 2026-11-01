# Protocol

What is measured, how, and in which order, for the campaign of 2026-11-01 (article #004) and the three campaigns it builds on. The raw logs of each are in `raw/`, one folder per campaign, each with its own `README.md` written during the campaign (in French).

## Machines

| label | machine | used for |
|---|---|---|
| `m3ultra` | a Mac Studio M3 Ultra, 96 GB, macOS 26.5 | every table of the article |
| `m1air` | a MacBook Air M1 (8 GB) | the f16 engine, the long prompts (0.8B, 2B) |
| `m5air` | a MacBook Air M5 (32 GB) | the attention kernels only (`2026-10-31-metal-f16-sdpa-long`); indicative, the machine heats and slows down within a minute of load |

Before each engine, the driver logs `pmset -g therm`, the thermal pressure level, the processes above 20 % CPU and `vm_stat`; again after it. On the MacBook Airs the runs are under `caffeinate -i -s` with pauses of 60 s; on the M5 the driver also waits (up to 10 min) for the thermal pressure level to come back to 0 and for no process to use more than 30 % CPU, and logs the waits.

## Engines

| engine | version | input |
|---|---|---|
| herbert-metal-bf16 | development commits named in each folder; the article's numbers: `f07e234`, the tree of [herbert-metal-bf16](https://github.com/xigh/herbert-metal-bf16) `v0.1.0` | the Hugging Face checkpoint, bf16 (`target/release`) or converted to f16 at load (`--features f16`, `target/f16`) |
| mlx-lm | 0.32.3 (2026-11-01), 0.32.0 (2026-10-29) | the checkpoint, bf16 (`mlx_lm benchmark`) |
| llama.cpp | `fb4b2737`, Metal | GGUF BF16 and F16 (`llama-bench`) |
| Rapid-MLX | 0.16.10 (2026-10-29 and 2026-10-30 only) | the checkpoint, through `_rapid_client.py` |
| BaseRT | 0.4.2 (2026-10-29 and 2026-10-30 only) | a bf16 `.base` converted from the checkpoint |

The engine's development tree was named `herbert-metal-f16` (its bench names its f16 build `herbert-metal-f16` in its tables, its bf16 build `herbert-metal-bf16`); in paths it is replaced by the public name.

## Speed: one model, its engines in turn (`_f16-campaign.sh`)

- **ppN**: N pseudo-random token ids from an empty context, the whole prefill timed (herbert-metal-bench's fixed linear congruential generator, llama-bench's random tokens, `mlx_lm benchmark`'s random ints, BaseRT `--fill random`, Rapid-MLX: the LCG ids decoded to text).
- **tgN**: N tokens from an (almost) empty context, each the argmax of the one before. With `DEPTH=D`: N tokens after an untimed prefill of D tokens (herbert `-d D`, llama-bench `-p 0 -d D`, `mlx_lm benchmark -p D`, whose generation follows its prompt).
- `W` warm-up runs (the long prompts: none, a run of 80k tokens being its own warm-up; `mlx_lm benchmark` always runs one), then `REPS` timed runs; `PAUSE` seconds between two engines.
- The order of the engines is reversed from one cell (model, length) to the next.
- What is read (`_art004_tables.py`): for herbert, the last `gpu t/s` table of `herbert-metal-bench` (the GPU time of the command buffers, `GPUEndTime − GPUStartTime` summed); its wall-clock table, printed just before in the same log, is within 0.5 % of it for the prefill and 1 to 12 % lower for the decode (largest on the 0.8B). For llama.cpp, `llama-bench`'s row (mean ± standard deviation of its runs). For mlx-lm, the `Averages` line of `mlx_lm benchmark`.

## Correctness: the same tokens (`bench/compare.py` of the engine)

Before speed, tokens. For each model, two prompts (`short`: « Bonjour, que fais-tu ? »; `long`: `bench/reference/long-prompt.txt`, 146 ids with the template), the template with `--no-system` and the thinking open, 48 greedy tokens. The prompt ids must be the reference's; the generated ids are compared with mlx-lm's (bf16 checkpoint, or the bf16 parameters cast to float16 after loading for the f16 engine) until the first difference. At that difference the top-5 logits of both are printed, and the divergence is classed as an exact tie of the reference (its two best tokens at the same bf16 logit), one bf16 ulp (the reference separates them by its resolution at that height, herbert puts them level or the other way), or a computation error (anything else).

## The load-time measure and its check

herbert's defaults (`gemm.auto=1`, `sdpa.auto=1`, `dec.auto=1`) time the candidate kernels of each shape at load and keep the fastest. From `2b86c37` each candidate is first run against a kernel in production (the first candidate) and rejected if its output differs by more than 1/64 of the largest magnitude; from `f07e234` that run is made on the full block and on a partial one with dispatches regenerated for the row count, from a zeroed output. The article's herbert numbers are all measured at `f07e234`; nothing in the engine changed after them.

## The campaign of 2026-11-01, in order (`raw/2026-11-01-metal-f16-art004`)

All on `m3ultra`, the engine's defaults, no other inference engine running.

1. bf16 on long prompts: herbert bf16, llama.cpp bf16 and mlx-lm, Qwen3.5 0.8B, 2B, 4B, 9B at pp2048, 8192, 24576, 61440, and pp81920 for the 0.8B; 2 runs without warm-up, tg128 from an empty context in the same runs (`_art004-1.sh`).
2. llama.cpp f16 at pp24576 and pp61440.
3. tg128 after 32 768 tokens of context: herbert f16 and bf16, llama.cpp bf16 and f16, mlx-lm; 2 runs after a warm-up.
4. Qwen3.8-27B and Qwen3.6-35B-A3B: pp512, pp8192, tg128; 2 runs after a warm-up.
5. Ids against mlx-lm: herbert bf16 against the bf16 reference, herbert f16 against the float16 and the bf16 references, six Qwen, two prompts.
6. The pieces: the per-kernel profile of the 0.8B f16 at pp2048 and pp61440 (`HERBERT_METAL_PROFILE=1`), `--dump-msl` of the kernels emitted at load.
3 bis. Last, tg128 after 81 920 tokens of context, the five engines, 1 run after a warm-up.

Points 2 to 6 ran from `_art004-rest.sh`. At 02:00 point 5 showed 0/48 identical ids for both herbert engines at `9e80538`: the `matmul2d`-on-fragments kernels written for the M5 (`sdpa_nax`, `gemm_nax`, `gemm_nax_swiglu`) build on the M3 Ultra, are timed fastest there, are picked, and return wrong values. The load-time check was added (`2b86c37`) and herbert's measures of points 1, 3, 4, 5, 6 redone (`_art004-rerun.sh`, folders `1r`, `3r`, `4r`, `5r`, `6r`); at 04:05 point 5 still showed 0/48 on the long prompts of the 4B and 9B (one `gemm_nax` form passed the check on the output projection), the check was corrected (`f07e234`) and herbert's measures redone a second time, then point 3 bis for every engine. llama.cpp's and mlx-lm's measures are those of the first pass (their engines did not change). The measures at `9e80538` and `2b86c37` are kept (folders without `r`, and `*-2b86c37`).

Afterwards: the 9B F16 GGUF, missing, converted from the BF16 one and measured for points 2, 3 and 3 bis (`_art004-9bf16.sh`); mlx-lm on the 27B retried alone (`_art004-mlx27b.sh`, failed again); the profile with the kernels the defaults picked fixed, one prefill counted (`_art004-prof.sh`).
