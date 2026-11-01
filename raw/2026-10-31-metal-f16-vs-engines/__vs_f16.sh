#!/bin/zsh
# The f16 engine against the other engines (the earlier campaign's selection) on m3ultra: this tree in f16 and its bf16 witness
# (gemm.auto=1, the defaults), llama.cpp on the GGUF in bf16 and in f16 (the f16 GGUF of S1: 0.8B, 2B, 4B; the others
# do not fit ${DATA_DIR:-$HOME}), mlx-lm and Rapid-MLX on the checkpoint, BaseRT on a bf16 .base (converted into ~ then
# dropped, neither on the 27B nor on the 35B), Gemma E2B, E4B and the six Qwen. bench/f16-campaign.sh: pp128,
# pp512, pp1024, pp2048, pp4096 (PP) and tg128, one warm-up, 3 runs, 30 s between two engines, in two rounds: the order turns from one model to the next
# and round B takes it reversed. Then tg128 after 4096 tokens of context, one round (herbert, llama.cpp, mlx-lm).
# usage: bench/_vs_f16.sh <outdir>
set -u
out=${1:A}
cd ${0:A:h}/..
export PP=${PP:-128,512,1024,2048,4096}
mkdir -p $out
date '+start %F %T'
git log --oneline -1; git status --short
shasum -a 256 target/f16/release/herbert-metal-bench target/release/herbert-metal-bench
(cd ~/llama.cpp-fb4b2737 && git log --oneline -1); ~/mlx-bench-venv/bin/python -c "import mlx_lm; print('mlx-lm', mlx_lm.__version__)"
${ENGINES_DIR:-$HOME/engines}/rapid-mlx/venv/bin/pip show rapid-mlx 2> /dev/null | grep Version
df -h ${DATA_DIR:-$HOME} ~ | tail -2
ps -axo pcpu,comm | awk 'NR>1 && $1>20'
C=bench/f16-campaign.sh
models=(E2B E4B Qwen3.5-0.8B Qwen3.5-2B Qwen3.5-4B Qwen3.5-9B Qwen3.8-27B Qwen3.6-35B-A3B)
engines_of() {
  case $1 in
    Qwen3.8-27B|Qwen3.6-35B-A3B) echo h-f16 h-bf16 llama-bf16 llama-f16 mlx rapid ;;
    *) echo h-f16 h-bf16 llama-bf16 llama-f16 mlx rapid basert ;;
  esac
}
for tour in A B; do
  echo "### tour $tour $(date +%T)"
  i=0
  for m in $models; do
    es=($(engines_of $m))
    k=$(( i % ${#es} ))
    es=(${es[$(( k + 1 )),-1]} ${es[1,$k]})
    [[ $tour == B ]] && es=(${(Oa)es})
    (( i++ ))
    $C $out/$tour $m $es
    sleep 30
  done
done
echo "### depth 4096 $(date +%T)"
i=0
for m in $models; do
  es=(h-f16 h-bf16 llama-bf16 llama-f16 mlx)
  k=$(( i % ${#es} ))
  es=(${es[$(( k + 1 )),-1]} ${es[1,$k]})
  (( i++ ))
  DEPTH=4096 $C $out/D $m $es
  sleep 30
done
df -h ${DATA_DIR:-$HOME} ~ | tail -2
date '+end %F %T'
