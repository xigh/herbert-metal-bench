#!/bin/zsh
# Why the prefill falls behind past ~2 000 tokens: one model, a few prompt lengths up to
# ~60k, and at each length the prefill with its methods forced — the attention tiles (sdpa.bq × sdpa.bk), the row-by-row
# attention (sdpa.prefill=0, up to ROWS_MAX: its scores buffer is block × heads × capacity × 4 bytes), the GEMMs on the
# SIMT kernel instead of the matrix units (gemm.mma=0), the block of rows (prefill.block) — and, first, the share of each
# kernel in the prefill (HERBERT_METAL_PROFILE=1, one dispatch per command buffer, base tune). ppN only (-n 0), one
# length per invocation (the capacity follows it), R runs after W warm-ups up to 8192 tokens, R_LONG and W_LONG above.
# PAUSE s between two runs (60 on the M1, under caffeinate -i -s), pmset -g therm and the top processes before each.
# The attention on the matrix units (sdpa_mma, cd5988f): mma (sdpa.prefill=2, K and V loaded by each simdgroup),
# mmast (=3, the tiles staged by the group), mmaqr (=3 with Q in registers), mmaqr8k (the same, prefill.block=8192);
# Metal 4's matmul2d (sdpa_mm2d): mm2d (sdpa.prefill=4, 64 queries x one head, 8 simdgroups), mm2dq32 (32 queries).
# usage: bench/_sdpa_long.sh <outdir> <model_dir> [bench binary]
#   PPS=2048,8192,24576,61440 VARIANTS="base q8k32 blk512 blk8192 simt rows" PROFILE=1 ROWS_MAX=8192
#   VARIANTS="base mma mmast mmaqr mmaqr8k" PROFILE=0: the attention kernels only
set -u
out=$1; model=$2; bin=${3:-$HOME/herbert-metal-bf16/target/f16/release/herbert-metal-bench}
PPS=${PPS:-2048,8192,24576,61440}
VARIANTS=${VARIANTS:-base q8k32 blk512 blk8192 simt rows}
PROFILE=${PROFILE:-1}
ROWS_MAX=${ROWS_MAX:-8192}
R=${R:-2}; W=${W:-1}; R_LONG=${R_LONG:-1}; W_LONG=${W_LONG:-0}
PAUSE=${PAUSE:-0}
mkdir -p $out
date '+start %F %T' | tee $out/_campaign.log
{ shasum -a 256 $bin; echo "model $model"; echo "PPS=$PPS VARIANTS=$VARIANTS PROFILE=$PROFILE ROWS_MAX=$ROWS_MAX R=$R W=$W R_LONG=$R_LONG W_LONG=$W_LONG PAUSE=$PAUSE"; sysctl -n machdep.cpu.brand_string; sw_vers -productVersion; } | tee -a $out/_campaign.log
tune_of() {  # variant -> --tune value ('' for the defaults)
  case $1 in
    base) echo "" ;;
    q8k32) echo "sdpa.bq=8,sdpa.bk=32" ;;
    blk512) echo "prefill.block=512" ;;
    blk8192) echo "prefill.block=8192" ;;
    simt) echo "gemm.mma=0" ;;
    rows) echo "sdpa.prefill=0" ;;
    mma) echo "sdpa.prefill=2" ;;
    mmast) echo "sdpa.prefill=3" ;;
    mmaqr) echo "sdpa.prefill=3,sdpa.qreg=1" ;;
    mmaqr8k) echo "sdpa.prefill=3,sdpa.qreg=1,prefill.block=8192" ;;
    mm2d) echo "sdpa.prefill=4" ;;
    mm2dq32) echo "sdpa.prefill=4,sdpa.mq=32" ;;
    *) echo "unknown variant $1" >&2; return 1 ;;
  esac
}
first=1
one() {  # variant, pp, log, env prefix...
  local v=$1 pp=$2 log=$3; shift 3
  local tune; tune=$(tune_of $v) || return
  local r=$R w=$W
  (( pp > 8192 )) && { r=$R_LONG; w=$W_LONG; }
  (( $# > 0 )) && { r=1; w=0; }  # the profile pass: one run, no warm-up
  [[ $first == 1 ]] || { (( PAUSE > 0 )) && { echo "pause $PAUSE s"; sleep $PAUSE; }; }
  first=0
  echo "=== $v pp$pp $(date '+%F %T') ${*:-}" | tee $log | tee -a $out/_campaign.log
  { pmset -g therm; top -l 1 -o cpu -n 4 -stats pid,command,cpu | tail -5; vm_stat | grep -E "Swapouts|Pages free"; } >> $log
  local args=($model -p $pp -n 0 -r $r -w $w)
  [[ -n $tune ]] && args+=(--tune $tune)
  echo "$* $bin $args" >> $log
  env "$@" $bin $args >> $log 2>&1
  echo "exit $? $(date '+%F %T')" >> $log
  { pmset -g therm; vm_stat | grep -E "Swapouts|Pages free"; } >> $log
  grep -E "^\[pp|^\| .* pp" $log | tail -3 | tee -a $out/_campaign.log
}
for pp in ${(s:,:)PPS}; do
  (( PROFILE )) && one base $pp $out/profile-pp$pp.log HERBERT_METAL_PROFILE=1
  for v in ${(s: :)VARIANTS}; do
    [[ $v == rows ]] && (( pp > ROWS_MAX )) && { echo "skip rows pp$pp (> ROWS_MAX $ROWS_MAX)" | tee -a $out/_campaign.log; continue; }
    one $v $pp $out/$v-pp$pp.log
  done
done
date '+end %F %T' | tee -a $out/_campaign.log
