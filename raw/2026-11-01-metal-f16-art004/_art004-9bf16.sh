#!/bin/zsh
# Article #004 : le GGUF F16 du 9B n'existe pas sur ${DATA_DIR:-$HOME} (plein à 100 %) : llama.cpp f16 y manque aux points 2 et 3.
# Après la fin des points 2 à 6 : le 9B BF16 converti en F16 par llama-quantize dans ~/gguf-f16 (18 Go ; 31 Go libres), les
# autres GGUF liés à côté, puis les cellules 9B llama-f16 des points 2, 3 et 3 bis avec GGUF=~/gguf-f16.
cd ~/herbert-metal-bf16
while pgrep -f "[_]art004-rest.sh" > /dev/null; do sleep 60; done
A=runs/art004
echo "start 9B f16 $(date '+%F %T')"
mkdir -p ~/gguf-f16
for g in ${GGUF_DIR:-$HOME/gguf}/*.gguf; do [[ -e ~/gguf-f16/${g:t} ]] || ln -s $g ~/gguf-f16/${g:t}; done
if [[ ! -s ~/gguf-f16/Qwen3.5-9B-F16.gguf ]]; then
  rm -f ~/gguf-f16/Qwen3.5-9B-F16.gguf
  DYLD_LIBRARY_PATH=~/llama.cpp-fb4b2737/build/bin ~/llama.cpp-fb4b2737/build/bin/llama-quantize ${GGUF_DIR:-$HOME/gguf}/Qwen3.5-9B-BF16.gguf ~/gguf-f16/Qwen3.5-9B-F16.gguf F16 > $A/9B-f16-quantize.log 2>&1
  echo "quantize exit $? $(date +%T)"; ls -la ~/gguf-f16/Qwen3.5-9B-F16.gguf
fi
echo "### 2 Qwen3.5-9B llama-f16 pp24576,61440 $(date +%T)"
GGUF=~/gguf-f16 PP=24576,61440 REPS=2 W=0 PAUSE=30 bench/f16-campaign.sh $A/2-f16-long Qwen3.5-9B llama-f16
echo "### 3 Qwen3.5-9B tg128 @ d32768 llama-f16 $(date +%T)"
GGUF=~/gguf-f16 DEPTH=32768 TG=128 REPS=2 W=1 PAUSE=30 bench/f16-campaign.sh $A/3-d32768 Qwen3.5-9B llama-f16
echo "### 3b Qwen3.5-9B tg128 @ d81920 llama-f16 $(date +%T)"
GGUF=~/gguf-f16 DEPTH=81920 TG=128 REPS=1 W=1 PAUSE=30 bench/f16-campaign.sh $A/3-d81920 Qwen3.5-9B llama-f16
echo "done 9B f16 $(date '+%F %T')"
