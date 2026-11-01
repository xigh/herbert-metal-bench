#!/bin/zsh
# The long prompts after bench/_vs_f16*.sh, on the machine it runs on: pp8192, pp16384, pp32768, pp65536, pp81920
# (PP_LONG) by bench/f16-campaign.sh, 2 runs and no warm-up (W=0: a run of 80k tokens is its own warm-up; mlx_lm
# benchmark still runs one), then tg128 after 81920 tokens of context (DEPTH_LONG), 1 run. The machine's paths and
# rules (GGUF, LLAMA, QUIET, PAUSE) from the environment, as the driver before it sets them. An engine that cannot hold
# the length fails in its log and the next one runs.
# LONG_ORDER=pp: one length at a time, every model and engine at that length before the next one (out/L/ppN/), so
# the comparison at the shorter lengths is complete first; the tgTG of mlx-lm after the first length only (TGP).
# usage: bench/_vs_f16_long.sh <outdir> MODEL:ENGINE,ENGINE,… …
set -u
out=${1:A}; shift
cd ${0:A:h}/..
mkdir -p $out
date '+start long %F %T'
C=bench/f16-campaign.sh
pl=${PP_LONG:-8192,16384,32768,65536,81920}
if [[ ${LONG_ORDER:-model} == pp ]]; then
  for n in ${(s:,:)pl}; do
    echo "### pp$n $(date +%T)"
    for p in "$@"; do
      PP=$n TGP=${pl%%,*} REPS=${REPS_LONG:-2} W=0 $C $out/L/pp$n ${p%%:*} ${(s:,:)${p#*:}}
      sleep ${PAUSE:-30}
    done
  done
else
  for p in "$@"; do
    PP=$pl REPS=${REPS_LONG:-2} W=0 $C $out/L ${p%%:*} ${(s:,:)${p#*:}}
    sleep ${PAUSE:-30}
  done
fi
echo "### depth ${DEPTH_LONG:-81920} $(date +%T)"
for p in "$@"; do
  DEPTH=${DEPTH_LONG:-81920} REPS=1 W=0 $C $out/DL ${p%%:*} ${(s:,:)${p#*:}}
  sleep ${PAUSE:-30}
done
date '+end long %F %T'
