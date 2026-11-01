#!/bin/zsh
# Article #004, 02 h 35 : les noyaux nax (sdpa_nax, gemm_nax) rendaient des valeurs fausses sur le M3 Ultra et les deux moteurs
# sortaient des ids faux au commit 9e80538 ; le commit 2b86c37 vérifie chaque candidat au chargement et les écarte. Les mesures
# herbert sont refaites à ce commit ; llama.cpp et mlx-lm du premier passage restent. Puis le point 3 bis complet.
cd ~/herbert-metal-bf16
A=runs/art004
echo "start rerun $(date '+%F %T') $(cat DEPLOYED_COMMIT)"
Q=(Qwen3.5-0.8B Qwen3.5-2B Qwen3.5-4B Qwen3.5-9B)
for pp in 2048 8192 24576 61440; do
  for m in $Q; do
    echo "### 1r $m pp$pp h-bf16 $(date +%T)"
    PP=$pp REPS=2 W=0 PAUSE=10 bench/f16-campaign.sh $A/1r-bf16-long/pp$pp $m h-bf16
  done
done
echo "### 1r Qwen3.5-0.8B pp81920 h-bf16 $(date +%T)"
PP=81920 REPS=2 W=0 PAUSE=10 bench/f16-campaign.sh $A/1r-bf16-long/pp81920 Qwen3.5-0.8B h-bf16
for m in $Q; do
  echo "### 3r $m tg128 @ d32768 h-f16 h-bf16 $(date +%T)"
  DEPTH=32768 TG=128 REPS=2 W=1 PAUSE=10 bench/f16-campaign.sh $A/3r-d32768 $m h-f16 h-bf16
done
for m in Qwen3.8-27B Qwen3.6-35B-A3B; do
  echo "### 4r $m pp512,8192 tg128 h-f16 h-bf16 $(date +%T)"
  PP=512,8192 TG=128 REPS=2 W=1 PAUSE=10 bench/f16-campaign.sh $A/4r-big $m h-f16 h-bf16
done
echo "### 5r ids $(date +%T)"
mkdir -p $A/5r-ids
python3 -I bench/compare.py target/release/herbert-metal-chat ~/models Qwen3.5-0.8B Qwen3.5-2B Qwen3.5-4B Qwen3.5-9B --log $A/5r-ids/bf16-vs-bf16 > $A/5r-ids/bf16-vs-bf16.out 2>&1
python3 -I bench/compare.py target/release/herbert-metal-chat ${MODELS_DIR:-$HOME/models} Qwen3.8-27B Qwen3.6-35B-A3B --log $A/5r-ids/bf16-vs-bf16 >> $A/5r-ids/bf16-vs-bf16.out 2>&1
python3 -I bench/compare.py target/f16/release/herbert-metal-chat ~/models Qwen3.5-0.8B Qwen3.5-2B Qwen3.5-4B Qwen3.5-9B --ref f16 --log $A/5r-ids/f16-vs-f16 > $A/5r-ids/f16-vs-f16.out 2>&1
python3 -I bench/compare.py target/f16/release/herbert-metal-chat ${MODELS_DIR:-$HOME/models} Qwen3.8-27B Qwen3.6-35B-A3B --ref f16 --log $A/5r-ids/f16-vs-f16 >> $A/5r-ids/f16-vs-f16.out 2>&1
python3 -I bench/compare.py target/f16/release/herbert-metal-chat ~/models Qwen3.5-0.8B Qwen3.5-2B Qwen3.5-4B Qwen3.5-9B --log $A/5r-ids/f16-vs-bf16 > $A/5r-ids/f16-vs-bf16.out 2>&1
python3 -I bench/compare.py target/f16/release/herbert-metal-chat ${MODELS_DIR:-$HOME/models} Qwen3.8-27B Qwen3.6-35B-A3B --log $A/5r-ids/f16-vs-bf16 >> $A/5r-ids/f16-vs-bf16.out 2>&1
echo "### 6r pieces $(date +%T)"
mkdir -p $A/6r-pieces
for pp in 2048 61440; do
  HERBERT_METAL_PROFILE=1 target/f16/release/herbert-metal-bench ~/models/qwen/Qwen3.5-0.8B -p $pp -n 0 -r 1 -w 0 > $A/6r-pieces/profile-0.8B-pp$pp.log 2>&1
done
target/f16/release/herbert-metal-bench ~/models/qwen/Qwen3.5-0.8B -p 512 -n 0 -r 1 -w 0 --dump-msl $A/6r-pieces/msl > $A/6r-pieces/dump-msl.log 2>&1
k=0
for m in $Q; do
  order=(h-f16 h-bf16 llama-bf16 llama-f16 mlx); (( k % 2 )) && order=(mlx llama-f16 llama-bf16 h-bf16 h-f16); (( k++ ))
  echo "### 3b $m tg128 @ d81920 $order $(date +%T)"
  GGUF=~/gguf-f16 DEPTH=81920 TG=128 REPS=1 W=1 PAUSE=20 bench/f16-campaign.sh $A/3-d81920 $m $order
done
echo "done rerun $(date '+%F %T')"
