#!/bin/zsh
# Article #004, point 4 : mlx-lm a échoué au chargement du 27B bf16 (« Command buffer execution failed: Caused GPU Timeout
# Error » dans mx.eval(model.parameters()), 01 h 36, juste après deux chargements de 54 Go par herbert). Un second essai,
# seul sur la machine, après tout le reste (le 35B-A3B aussi s'il a échoué).
cd ${HERBERT_DIR:-$HOME/herbert-metal-bf16}
while pgrep -f "[_]art004-(rest|9bf16).sh" > /dev/null; do sleep 60; done
A=runs/art004
echo "start mlx retry $(date '+%F %T')"
for m in Qwen3.8-27B Qwen3.6-35B-A3B; do
  if ! grep -q "Averages" $A/4-big/$m/mlx.log 2>/dev/null; then
    mv $A/4-big/$m/mlx.log $A/4-big/$m/mlx-failed-1.log 2>/dev/null
    echo "### 4 retry $m mlx $(date +%T)"
    PP=512,8192 TG=128 REPS=2 W=1 PAUSE=30 bench/f16-campaign.sh $A/4-big $m mlx
  fi
done
echo "done mlx retry $(date '+%F %T')"
