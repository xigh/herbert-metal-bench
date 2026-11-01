#!/bin/zsh
# Point 6, complément : le profil par noyau (HERBERT_METAL_PROFILE=1) compte aussi les dispatchs du réglage au chargement ;
# pour une part de l'attention propre, les mêmes noyaux que ceux que les défauts ont choisis sur le M3 Ultra (gemm_mm2d t32,
# sdpa_mma étagé + Q en registres), figés : pas de mesure au chargement, un seul prefill compté. Après la fin du rerun.
cd ${HERBERT_DIR:-$HOME/herbert-metal-bf16}
while pgrep -f "[_]art004-rerun.sh" > /dev/null; do sleep 30; done
A=runs/art004/6r-pieces
for pp in 2048 61440; do
  HERBERT_METAL_PROFILE=1 target/f16/release/herbert-metal-bench ~/models/qwen/Qwen3.5-0.8B -p $pp -n 0 -r 1 -w 0 --tune gemm.mma=2,gemm.tn=32,sdpa.prefill=3,sdpa.qreg=1,prefill.block=$(( pp < 8192 ? pp : 8192 )) > $A/profile-fixed-0.8B-pp$pp.log 2>&1
done
echo "done prof $(date '+%F %T')"
