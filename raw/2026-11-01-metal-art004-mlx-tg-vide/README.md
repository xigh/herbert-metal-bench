# 2026-11-01 — article #004, complément : le décodage de mlx-lm à contexte presque vide, encadré par herbert et llama.cpp (six Qwen bf16, M3 Ultra)

La question (2026-11-01, 09 h) : dans les tableaux de décodage de l'article, herbert (`herbert-metal-bench` tg128) et llama.cpp (`llama-bench` tg128) décodent 128 tokens depuis un contexte vide, mlx-lm vient du point 1 de `2026-11-01-metal-f16-art004` (`mlx_lm benchmark -p 2048 -g 128`), donc ses 128 tokens suivent un prompt de 2 048 tokens, et pour le 35B-A3B la fourchette « 73 à 78 » mélange deux longueurs de prompt (pp512 et pp8192, `4-big`). Mesurer mlx-lm au plus près d'un contexte vide pour comparer à contexte égal.

Machine et versions : Mac Studio M3 Ultra (m3ultra), herbert-metal-f16 à `f07e234` (le commit proposé pour le tag, bf16 `target/release`, réglages par défaut), mlx-lm 0.32.0 sur mlx 0.32.3 (le « mlx-lm 0.32.3 » de `2026-11-01-metal-f16-art004` était la version de mlx), llama.cpp `fb4b2737` (`~/llama.cpp-fb4b2737`, GGUF BF16 de `$GGUF_DIR`). Aucun autre moteur LLM ne tournait. Lancée 09 h 04, finie 09 h 33, relance du dernier run 09 h 34.

## Protocole

Pour chaque modèle (Qwen3.5 0.8B, 2B, 4B, 9B, Qwen3.8-27B, Qwen3.6-35B-A3B), dans la même session et à la suite, pause de 30 s entre deux moteurs, l'ordre inversé d'un modèle au suivant (`_art004-mlxtg.sh`, sortie `_mlxtg.out`) :

- `mlx-p1` : `python -m mlx_lm benchmark --model <checkpoint bf16> -p 1 -g 128 -n 2` (un échauffement puis deux essais, la ligne `Averages`) ;
- `mlx-p16` : la même à `-p 16` ;
- `h-bf16` : `herbert-metal-bench <checkpoint> -p 16 -n 128 -r 2` (un échauffement, deux runs ; le tg128 part d'un contexte vide : `reset` puis génération depuis un premier token, la mesure pp16 est à part ; deux tables, temps réel puis temps GPU) ;
- `llama-bf16` : `llama-bench -m <GGUF BF16> -p 0 -n 128 -r 2 -o md`.

Le brief prévoyait un seul essai de mlx-lm sur le 27B (deux GPU Timeout au chargement la nuit) : il a chargé et tourné les deux fois. Le dernier run de la campagne, le `-p 1` du 35B-A3B, a échoué au chargement (`GPU Timeout` dans `mx.eval(model.parameters())`, le même point que le 27B la nuit) 30 s après un `-p 16` qui avait chargé ; relancé une fois, seul, 47 s plus tard : 74,0 (`mlx-p1-retry.log`). Les deux logs sont gardés.

## Dossiers

- `logs/<modèle>/{mlx-p1,mlx-p16,h-bf16,llama-bf16}.log` : chaque run entier (état thermique, mémoire et processus avant et après, sortie complète du moteur, code de sortie) ; `logs/Qwen3.6-35B-A3B/mlx-p1-retry.log` : la relance.
- `_mlxtg.out` : la sortie du pilote (`_art004-mlxtg.sh`).
- `_table.md` : la table, par `_mlxtg_table.py` (= `bench/mlxtg_table.py` du dépôt).

## Résultat : tg128, tokens/s

| modèle | herbert bf16 (réel) | herbert bf16 (GPU) | mlx-lm -p 1 | mlx-lm -p 16 | llama.cpp bf16 | mlx-lm publié (pp2048) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 336,3 ± 0,2 | 383,1 ± 0,2 | 255,6 | 255,1 | 207,3 ± 0,3 | 250 |
| Qwen3.5-2B | 170,4 ± 0,1 | 181,6 ± 0,1 | 144,2 | 144,2 | 133,3 ± 0,7 | 144 |
| Qwen3.5-4B | 82,5 ± 0,0 | 85,3 ± 0,0 | 71,2 | 71,4 | 68,1 ± 0,1 | 71 |
| Qwen3.5-9B | 46,3 ± 0,0 | 47,1 ± 0,0 | 40,9 | 40,9 | 40,8 ± 0,2 | 41 |
| Qwen3.8-27B | 14,7 ± 0,0 | 14,8 ± 0,0 | 13,4 | 13,4 | 13,4 ± 0,0 | ne charge pas |
| Qwen3.6-35B-A3B | 86,6 ± 0,0 | 90,3 ± 0,0 | GPU Timeout au chargement ; relancé : 74,0 | 73,7 | 71,9 ± 0,0 | 73 à 78 |

Lecture : le contexte vide ne change pas le décodage de mlx-lm, ses valeurs à `-p 1` et `-p 16` sont celles publiées après 2 048 tokens à ±2 % (0.8B 250 → 256, 2B 144 → 144, 4B 71 → 71, 9B 41 → 41) ; le 35B-A3B à contexte vide fait 73,7–74,0, le bas de la fourchette « 73 à 78 » (le 78 venait du pp8192) ; le 27B a chargé cette fois : 13,4, llama.cpp 13,4, herbert 14,7 réel. mlx-lm ne passe devant herbert nulle part : herbert en temps réel est à ×1,32 (0.8B), ×1,18 (2B), ×1,16 (4B), ×1,13 (9B), ×1,10 (27B), ×1,17 (35B-A3B) de mlx-lm, et son temps réel est en dessous de son temps GPU de 12 % sur le 0.8B, 6 % sur le 2B, 3 % sur le 4B, 2 % sur le 9B, 1 % sur le 27B, 4 % sur le 35B-A3B. llama.cpp bf16 à `-p 0` est à ×0,81–1,00 de mlx-lm (égal sur le 9B et le 27B). Les herbert de ce dossier (réel 336 / 170 / 82,5 / 46,3 / 14,7 / 86,6) sont au niveau de ceux de `2026-11-01-metal-f16-art004` (`4-big` 27B 14,9 et 35B 91,0 en GPU : ici 14,8 et 90,3).
