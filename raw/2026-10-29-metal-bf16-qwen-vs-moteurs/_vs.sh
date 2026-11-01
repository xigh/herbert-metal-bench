#!/bin/zsh
# herbert-metal-bf16 (gemm.auto=1, the defaults) contre llama.cpp, mlx-lm, Rapid-MLX et BaseRT sur m3ultra, le protocole
# de la campagne précédente (bench/qwen-campaign.sh : pp512 puis tg128, un échauffement, 3 runs, 30 s entre deux
# moteurs), en deux tours : l'ordre des moteurs tourne d'un modèle à l'autre, et le tour B le prend à l'envers.
# BaseRT ni sur le 27B (pagination) ni sur le 35B (ne tient pas) ; ses .base gardés entre les deux tours
# et supprimés après le tour B, celui du 0.8B gardé comme avant.  usage : _vs.sh <outdir>
out=$1
cd ${0:A:h}/..
mkdir -p $out
date '+start %F %T'
git log --oneline -1 2>/dev/null; shasum -a 256 target/release/herbert-metal-chat target/release/herbert-metal-bench
df -h ${DATA_DIR:-$HOME} | tail -1
echo "### ids, réglages par défaut $(date +%T)"
python3 -I bench/compare.py target/release/herbert-metal-chat ~/models Qwen3.5-0.8B Qwen3.5-2B Qwen3.5-4B Qwen3.5-9B --log $out/ids > $out/ids.txt 2>&1
python3 -I bench/compare.py target/release/herbert-metal-chat ${MODELS_DIR:-$HOME/models} Qwen3.8-27B Qwen3.6-35B-A3B --log $out/ids >> $out/ids.txt 2>&1
cat $out/ids.txt
C=bench/qwen-campaign.sh
for tour in A B; do
  echo "### tour $tour $(date +%T)"
  rev=0; [[ $tour == B ]] && rev=1
  run() {  # modèle, moteurs dans l'ordre du tour A
    local m=$1; shift
    local es=("$@")
    [[ $rev == 1 ]] && es=(${(Oa)es})
    $C $out/$tour $m $es
  }
  run Qwen3.5-0.8B herbert llama mlx rapid basert
  run Qwen3.5-2B llama mlx rapid basert herbert
  run Qwen3.5-4B mlx rapid basert herbert llama
  run Qwen3.5-9B rapid basert herbert llama mlx
  run Qwen3.8-27B herbert llama mlx rapid
  run Qwen3.6-35B-A3B llama mlx rapid herbert
done
for m in Qwen3.5-2B Qwen3.5-4B Qwen3.5-9B; do
  rm -f ${ENGINES_DIR:-$HOME/engines}/baseRT/models/$m-bf16.base && echo "removed $m-bf16.base"
done
df -h ${DATA_DIR:-$HOME} | tail -1
date '+end %F %T'
