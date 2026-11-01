#!/bin/zsh
# Article #004, points 2 à 6 (2026-10-31, 23 h 45), à la suite du point 1 (_art004-1.sh). M3 Ultra, arbre 9e80538, défauts du moteur.
cd ${HERBERT_DIR:-$HOME/herbert-metal-bf16}
while pgrep -f "[_]art004-1.sh" > /dev/null; do sleep 30; done
A=runs/art004
echo "start rest $(date '+%F %T')"
Q=(Qwen3.5-0.8B Qwen3.5-2B Qwen3.5-4B Qwen3.5-9B)
# 2. Les valeurs approchées de la campagne 2026-10-31-metal-f16-sdpa-long : llama.cpp f16 à pp24576 et pp61440 (mlx-lm y est mesuré au point 1, même moteur, mêmes longueurs).
for m in $Q; do
  echo "### 2 $m llama-f16 pp24576,61440 $(date +%T)"
  PP=24576,61440 REPS=2 W=0 PAUSE=30 bench/f16-campaign.sh $A/2-f16-long $m llama-f16
done
# 3. tg128 après 32 768 tokens : herbert f16 et bf16, llama.cpp (bf16 et f16), mlx-lm ; ordre inversé d'un modèle au suivant.
k=0
for m in $Q; do
  order=(h-f16 h-bf16 llama-bf16 llama-f16 mlx); (( k % 2 )) && order=(mlx llama-f16 llama-bf16 h-bf16 h-f16); (( k++ ))
  echo "### 3 $m tg128 @ d32768 $order $(date +%T)"
  DEPTH=32768 TG=128 REPS=2 W=1 PAUSE=30 bench/f16-campaign.sh $A/3-d32768 $m $order
done
# 4. Le 27B et le 35B-A3B : pp512, pp8192, tg128.
for m in Qwen3.8-27B Qwen3.6-35B-A3B; do
  echo "### 4 $m pp512,8192 tg128 $(date +%T)"
  PP=512,8192 TG=128 REPS=2 W=1 PAUSE=30 bench/f16-campaign.sh $A/4-big $m h-f16 h-bf16 llama-bf16 mlx
done
# 5. Les ids contre mlx-lm : herbert bf16 contre la référence bf16, herbert f16 contre la référence float16 et contre la bf16.
echo "### 5 ids $(date +%T)"
mkdir -p $A/5-ids
python3 -I bench/compare.py target/release/herbert-metal-chat ~/models Qwen3.5-0.8B Qwen3.5-2B Qwen3.5-4B Qwen3.5-9B --log $A/5-ids/bf16-vs-bf16 > $A/5-ids/bf16-vs-bf16.out 2>&1
python3 -I bench/compare.py target/release/herbert-metal-chat ${MODELS_DIR:-$HOME/models} Qwen3.8-27B Qwen3.6-35B-A3B --log $A/5-ids/bf16-vs-bf16 >> $A/5-ids/bf16-vs-bf16.out 2>&1
python3 -I bench/compare.py target/f16/release/herbert-metal-chat ~/models Qwen3.5-0.8B Qwen3.5-2B Qwen3.5-4B Qwen3.5-9B --ref f16 --log $A/5-ids/f16-vs-f16 > $A/5-ids/f16-vs-f16.out 2>&1
python3 -I bench/compare.py target/f16/release/herbert-metal-chat ${MODELS_DIR:-$HOME/models} Qwen3.8-27B Qwen3.6-35B-A3B --ref f16 --log $A/5-ids/f16-vs-f16 >> $A/5-ids/f16-vs-f16.out 2>&1
python3 -I bench/compare.py target/f16/release/herbert-metal-chat ~/models Qwen3.5-0.8B Qwen3.5-2B Qwen3.5-4B Qwen3.5-9B --log $A/5-ids/f16-vs-bf16 > $A/5-ids/f16-vs-bf16.out 2>&1
python3 -I bench/compare.py target/f16/release/herbert-metal-chat ${MODELS_DIR:-$HOME/models} Qwen3.8-27B Qwen3.6-35B-A3B --log $A/5-ids/f16-vs-bf16 >> $A/5-ids/f16-vs-bf16.out 2>&1
# 6. Les pièces : la part de l'attention (profil par noyau, 0.8B f16, pp2048 et pp61440), la ligne [autotune] (dans tout log de
#    chargement), l'extrait --dump-msl (les noyaux émis au chargement, sdpa_mma parmi eux).
echo "### 6 pieces $(date +%T)"
mkdir -p $A/6-pieces
for pp in 2048 61440; do
  HERBERT_METAL_PROFILE=1 target/f16/release/herbert-metal-bench ~/models/qwen/Qwen3.5-0.8B -p $pp -n 0 -r 1 -w 0 > $A/6-pieces/profile-0.8B-pp$pp.log 2>&1
done
target/f16/release/herbert-metal-bench ~/models/qwen/Qwen3.5-0.8B -p 512 -n 0 -r 1 -w 0 --dump-msl $A/6-pieces/msl > $A/6-pieces/dump-msl.log 2>&1
# 3 bis. tg128 après 81 920 tokens, si le temps le permet (en dernier).
k=0
for m in $Q; do
  order=(h-f16 h-bf16 llama-bf16 llama-f16 mlx); (( k % 2 )) && order=(mlx llama-f16 llama-bf16 h-bf16 h-f16); (( k++ ))
  echo "### 3b $m tg128 @ d81920 $order $(date +%T)"
  DEPTH=81920 TG=128 REPS=1 W=1 PAUSE=30 bench/f16-campaign.sh $A/3-d81920 $m $order
done
echo "done rest $(date '+%F %T')"
