#!/bin/zsh
# The f16 engine against llama.cpp on the M1 Air (mlx-lm is not installed there): this tree in f16 and its bf16 witness
# (the defaults), llama.cpp on the GGUF in bf16 and in f16 (~/s1-f16/gguf), Qwen3.5-0.8B and 2B, then Gemma E2B (herbert
# only: its bf16 GGUF does not fit 8 GB). bench/f16-campaign.sh: pp128, pp512, pp1024, pp2048, pp4096 (PP) and tg128, one warm-up, 5 runs, 60 s between
# two engines, pmset -g therm before and after each. Series A, then series B one hour after A's start with every order
# reversed, then tg128 after 4096 tokens of context (one series). Under caffeinate -i -s.
# usage: caffeinate -i -s bench/_vs_f16_m1air.sh <outdir>   env: LLAMA (llama-bench's dir)
set -u
out=${1:A}
cd ${0:A:h}/..
export PP=${PP:-128,512,1024,2048,4096} REPS=${REPS:-5} PAUSE=${PAUSE:-60} GGUF=$HOME/s1-f16/gguf LLAMA=${LLAMA:-$HOME/llama.cpp-s1/build/bin}
mkdir -p $out
t0=$(date +%s)
date '+start %F %T'
git log --oneline -1; git status --short
shasum -a 256 target/f16/release/herbert-metal-bench target/release/herbert-metal-bench $LLAMA/llama-bench
(cd $LLAMA && git log --oneline -1 2> /dev/null)
pmset -g | grep -E " sleep"; pmset -g therm
C=bench/f16-campaign.sh
series() {
  local s=$1
  local plan=("Qwen3.5-0.8B h-f16 llama-bf16 h-bf16 llama-f16" "Qwen3.5-2B llama-f16 h-bf16 llama-bf16 h-f16" "E2B h-f16 h-bf16")
  [[ $s == B ]] && plan=(${(Oa)plan})
  for p in $plan; do
    local w=(${=p})
    local es=(${w[2,-1]})
    [[ $s == B ]] && es=(${(Oa)es})
    $C $out/$s $w[1] $es
    sleep $PAUSE
  done
}
series A
wait_s=$(( t0 + 3600 - $(date +%s) ))
(( wait_s > 0 )) && { echo "waiting $wait_s s for series B"; sleep $wait_s; }
series B
echo "### depth 4096 $(date +%T)"
DEPTH=4096 $C $out/D Qwen3.5-0.8B h-f16 llama-f16 h-bf16 llama-bf16; sleep $PAUSE
DEPTH=4096 $C $out/D Qwen3.5-2B llama-bf16 h-bf16 llama-f16 h-f16
pmset -g therm
date '+end %F %T'
