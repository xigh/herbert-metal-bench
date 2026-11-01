#!/bin/zsh
# One model of the Qwen bf16 speed campaign on m3ultra: pp512 and tg128 by each
# engine in the order given, a pause between two engines, every output kept.
#
# usage: bench/qwen-campaign.sh OUT_DIR NAME ENGINE...
#   NAME    Qwen3.5-0.8B | Qwen3.5-2B | Qwen3.5-4B | Qwen3.5-9B | Qwen3.8-27B | Qwen3.6-35B-A3B
#   ENGINE  herbert | llama | mlx | rapid | basert
# env: PAUSE (seconds, default 30), REPS (default 3), TUNE (herbert's --tune),
#      DROP_BASE=1 (delete the model's .base after BaseRT's run)
#
# The same protocol for all: pp512 from an empty context on pseudo-random
# ids (llama-bench's random tokens, BaseRT --fill random, mlx_lm benchmark's
# random ints, herbert-metal-bench's LCG, Rapid-MLX's decoded LCG text), tg128
# from an (almost) empty context, one warm-up, REPS timed runs.
set -u
out=$1; name=$2; shift 2
PAUSE=${PAUSE:-30}; REPS=${REPS:-3}; TUNE=${TUNE:-}
HB=$HOME/herbert-metal-bf16
EN=${ENGINES_DIR:-$HOME/engines}
case $name in
  Qwen3.5-*) dir=$HOME/models/qwen/$name ;;
  *) dir=${MODELS_DIR:-$HOME/models}/$name ;;
esac
gguf=${GGUF_DIR:-$HOME/gguf}/$name-BF16.gguf
base=$EN/baseRT/models/$name-bf16.base
mkdir -p $out/$name
first=1
for e in "$@"; do
  [[ $first == 1 ]] || { echo "pause $PAUSE s"; sleep $PAUSE; }
  first=0
  log=$out/$name/$e.log
  echo "=== $name $e $(date '+%F %T')" | tee $log
  vm_stat | grep -E "Swapouts|Pages free" >> $log
  case $e in
    herbert)
      $HB/target/release/herbert-metal-bench $dir -p 512 -n 128 -r $REPS -w 1 ${TUNE:+--tune $TUNE} >> $log 2>&1 ;;
    llama)
      $HOME/llama.cpp-fb4b2737/build/bin/llama-bench -m $gguf -p 512 -n 128 -r $REPS -o md >> $log 2>&1 ;;
    mlx)
      $HOME/mlx-bench-venv/bin/python -m mlx_lm benchmark --model $dir -p 512 -g 128 -n $REPS >> $log 2>&1 ;;
    rapid)
      RAPID_MLX_TELEMETRY=0 HF_HUB_OFFLINE=1 $EN/rapid-mlx/venv/bin/rapid-mlx serve $dir --disable-model-downloads --no-mllm \
        --disable-prefix-cache --disable-disk-caches --port 8765 > $out/$name/rapid-serve.log 2>&1 &
      pid=$!
      up=0
      for i in {1..600}; do
        curl -sf http://localhost:8765/v1/models > /dev/null && { up=1; break; }
        kill -0 $pid 2> /dev/null || break
        sleep 1
      done
      if [[ $up == 1 ]]; then
        $EN/rapid-mlx/venv/bin/python -I $HB/bench/rapid_client.py $dir 8765 -p 512 -n 128 -r $REPS -w 1 >> $log 2>&1
      else
        echo "rapid-mlx did not come up" >> $log
      fi
      kill $pid 2> /dev/null; wait $pid 2> /dev/null ;;
    basert)
      if [[ ! -f $base ]]; then
        (cd $EN/baseRT && ./base-convert/target/release/basert convert $dir --target bf16 -o $base) > $out/$name/basert-convert.log 2>&1
      fi
      (cd $EN/baseRT && ./build/basert-bench $base -p 512 -n 128 -r $REPS -w 1 --fill random) >> $log 2>&1
      [[ ${DROP_BASE:-0} == 1 ]] && rm -f $base && echo "removed $base" >> $log ;;
  esac
  echo "exit $? $(date '+%F %T')" >> $log
  vm_stat | grep -E "Swapouts|Pages free" >> $log
  grep -E "^\|" $log | tail -3
done
