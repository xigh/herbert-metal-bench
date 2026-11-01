#!/bin/zsh
# The mma attention (sdpa.prefill=3, sdpa.qreg=1) against the scalar flash on the M3 Ultra, the other Qwen: the ids
# against mlx-lm float16 (bench/compare.py), then pp2048, pp8192 and pp24576 of both (-n 0; 2 runs after a warm-up,
# 1 run without at 24576). usage: bench/_mma_m3ultra.sh <outdir> [models...]
set -u
out=$1; shift
models=("$@"); (( $#models )) || models=(Qwen3.5-2B Qwen3.5-4B Qwen3.5-9B)
HB=$HOME/herbert-metal-bf16
mkdir -p $out
date '+start %F %T' | tee $out/_campaign.log
shasum -a 256 $HB/target/f16/release/herbert-metal-bench | tee -a $out/_campaign.log
for m in $models; do
  echo "=== ids $m $(date '+%F %T')" | tee -a $out/_campaign.log
  python3 -I $HB/bench/compare.py $HB/target/f16/release/herbert-metal-chat $HOME/models $m --ref f16 --log $out/ids-$m -- --tune sdpa.prefill=3,sdpa.qreg=1 2>&1 | tee $out/ids-$m.log | tail -2 | tee -a $out/_campaign.log
  for v in base mmaqr; do
    tune=""; [[ $v == mmaqr ]] && tune="--tune sdpa.prefill=3,sdpa.qreg=1"
    for pp in 2048 8192 24576; do
      r=2; w=1; (( pp > 8192 )) && { r=1; w=0; }
      echo "=== $v $m pp$pp $(date '+%F %T')" | tee -a $out/_campaign.log
      $HB/target/f16/release/herbert-metal-bench $HOME/models/qwen/$m -p $pp -n 0 -r $r -w $w ${=tune} > $out/$v-$m-pp$pp.log 2>&1
      echo "exit $?" >> $out/$v-$m-pp$pp.log
      grep -E "^\[pp\S+\] run|error" $out/$v-$m-pp$pp.log | tail -2 | tee -a $out/_campaign.log
    done
  done
done
date '+end %F %T' | tee -a $out/_campaign.log
