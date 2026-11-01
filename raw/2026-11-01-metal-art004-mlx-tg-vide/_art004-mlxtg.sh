#!/bin/zsh
# Article #004, complément du 2026-11-01 (09 h) : le décodage de mlx-lm à contexte presque vide (benchmark -p 1 et
# -p 16, -g 128, -n 2) sur les six Qwen bf16, encadré dans la même session par herbert-metal-bench bf16 (-p 16 -n 128 -r 2,
# réglages par défaut) et llama-bench (-p 0 -n 128 -r 2, GGUF BF16). Ordre des moteurs inversé d'un modèle au suivant.
# M3 Ultra, arbre f07e234, mlx-lm 0.32.0 / mlx 0.32.3, llama.cpp fb4b2737. Le 27B : un essai de mlx-lm, pas plus.
set -u
cd ~/herbert-metal-bf16
A=runs/art004/mlxtg
PAUSE=30
LLAMA=$HOME/llama.cpp-fb4b2737/build/bin
MLX=$HOME/mlx-bench-venv/bin/python
echo "start mlxtg $(date '+%F %T')"
k=0
for m in Qwen3.5-0.8B Qwen3.5-2B Qwen3.5-4B Qwen3.5-9B Qwen3.8-27B Qwen3.6-35B-A3B; do
  case $m in
    Qwen3.8-27B|Qwen3.6-35B-A3B) dir=${MODELS_DIR:-$HOME/models}/$m ;;
    *) dir=$HOME/models/qwen/$m ;;
  esac
  gguf=${GGUF_DIR:-$HOME/gguf}/$m-BF16.gguf
  order=(mlx-p1 mlx-p16 h-bf16 llama-bf16); (( k % 2 )) && order=(llama-bf16 h-bf16 mlx-p16 mlx-p1); (( k++ ))
  echo "### $m $order $(date +%T)"
  mkdir -p $A/$m
  first=1
  for e in $order; do
    [[ $first == 1 ]] || sleep $PAUSE
    first=0
    log=$A/$m/$e.log
    echo "=== $m $e $(date '+%F %T')" | tee $log
    { pmset -g therm; notifyutil -g com.apple.system.thermalpressurelevel; ps -axo pcpu,comm | awk 'NR>1 && $1>20'; vm_stat | grep -E "Swapouts|Pages free"; } >> $log 2>&1
    case $e in
      mlx-p1)  echo "input $dir" >> $log; $MLX -m mlx_lm benchmark --model $dir -p 1 -g 128 -n 2 >> $log 2>&1 ;;
      mlx-p16) echo "input $dir" >> $log; $MLX -m mlx_lm benchmark --model $dir -p 16 -g 128 -n 2 >> $log 2>&1 ;;
      h-bf16)  echo "input $dir" >> $log; target/release/herbert-metal-bench $dir -p 16 -n 128 -r 2 >> $log 2>&1 ;;
      llama-bf16) echo "input $gguf" >> $log; DYLD_LIBRARY_PATH=$LLAMA $LLAMA/llama-bench -m $gguf -p 0 -n 128 -r 2 -o md >> $log 2>&1 ;;
    esac
    echo "exit $? $(date '+%F %T')" >> $log
    { pmset -g therm; notifyutil -g com.apple.system.thermalpressurelevel; vm_stat | grep -E "Swapouts|Pages free"; } >> $log 2>&1
    grep -E "^\||generation_tps|Error" $log | grep -v autotune | tail -4
  done
done
echo "done mlxtg $(date '+%F %T')"
