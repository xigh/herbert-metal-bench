#!/bin/zsh
# Article #004, point 1 (2026-10-31, 23 h 30) : herbert bf16 contre llama.cpp bf16 et mlx-lm bf16 sur le M3 Ultra,
# Qwen3.5 0.8B, 2B, 4B, 9B à pp2048, 8192, 24576, 61440 (pp81920 pour le 0.8B), protocole de la campagne du 2026-10-31 (bench/f16-campaign.sh,
# 2 runs sans échauffement), l'ordre des moteurs inversé d'une cellule à la suivante. Arbre 9e80538, défauts du moteur.
cd ${HERBERT_DIR:-$HOME/herbert-metal-bf16}
out=runs/art004/1-bf16-long; mkdir -p $out
echo "start $(date '+%F %T') $(cat DEPLOYED_COMMIT)"
k=0
for pp in 2048 8192 24576 61440; do
  for m in Qwen3.5-0.8B Qwen3.5-2B Qwen3.5-4B Qwen3.5-9B; do
    order=(h-bf16 llama-bf16 mlx); (( k % 2 )) && order=(mlx llama-bf16 h-bf16); (( k++ ))
    echo "### $m pp$pp $order $(date +%T)"
    PP=$pp REPS=2 W=0 PAUSE=30 bench/f16-campaign.sh $out/pp$pp $m $order
  done
done
echo "### Qwen3.5-0.8B pp81920 $(date +%T)"
PP=81920 REPS=2 W=0 PAUSE=30 bench/f16-campaign.sh $out/pp81920 Qwen3.5-0.8B h-bf16 llama-bf16 mlx
echo "done $(date '+%F %T')"
