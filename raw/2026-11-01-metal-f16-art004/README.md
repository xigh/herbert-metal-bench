# 2026-11-01 — herbert-metal-f16 sur le M3 Ultra au commit `9e80538` (proposé pour le tag public) : les mesures de l'article #004

> **Alerte, 02 h 00 : les mesures herbert de ce dossier faites au commit `9e80538` (points 1, 3, 4, 5, 6 : `logs/1-bf16-long`, `3-d32768`, `4-big`, `5-ids`) sont FAUSSES.** Le point 5 a montré 0/48 ids identiques (tokens répétés, marges de logits à 0,06) pour le témoin bf16, puis pour le moteur f16 (`logs/5-diag/`). Cause : les noyaux à fragments `matmul2d` écrits pour la M5 (`sdpa_nax`, `gemm_nax`, `gemm_nax_swiglu`, proposés à `sdpa.auto` et `gemm.auto` depuis `1319e53`) se construisent sur le M3 Ultra, y sont chronométrés plus vite que `sdpa_mma` et sont choisis, et y rendent des valeurs fausses (le placement des éléments dans les lanes est celui de la M5). Vérification : `sdpa.prefill=4,gemm.mma=1` rend 48/48 en bf16 et en f16, `sdpa.prefill=3` seul ou `gemm.mma=1` seul 0/48. **Correctif `1dcbd57` → `2b86c37`** : chaque candidat d'un choix au chargement est exécuté une fois contre le premier (un noyau en production : `flash`, `mma`, `apart`) sur un nombre de lignes qui n'est multiple d'aucune tuile, les lignes utiles comparées, les caches K/V remplis avant (à zéro tous les noyaux rendaient zéro et le contrôle ne voyait rien) ; un écart au-delà de 1/64 de la plus grande valeur écarte le candidat (`[autotune] … wrong against flash on 40 rows, never picked: nax`). Les deux moteurs rendent 48/48 sur les deux prompts du 0.8B avec le correctif (`5-diag/fixed3-*`) ; deux formes `gemm_nax` sur 2048 × 1024 passent le contrôle et restent prises. **Les mesures herbert sont refaites au commit `2b86c37`** (`logs/1r-bf16-long`, `3r-d32768`, `4r-big`, `5r-ids`, `6r-pieces`, `_art004-rerun.sh`, sortie `logs/rerun.out`) ; celles de llama.cpp et mlx-lm du premier passage restent valables. Le commit proposé pour le tag devient `2b86c37`, puis `f07e234` (second correctif).

> **Second correctif, 04 h 05 (`f07e234`, le commit proposé pour le tag)** : le point 5 refait au commit `2b86c37` rendait encore 0/48 sur les prompts longs du 4B et du 9B : une forme `gemm_nax` (64 × 64, bande 2) passait le contrôle sur la projection de sortie (4096 × 2560, 4096 × 4096) et y était prise. Le contrôle comparait sur un nombre partiel de lignes avec les groupes du bloc entier, ce qui faussait la comparaison (`mm2d` lui-même y différait de `mma`). Désormais chaque candidat est exécuté sur le bloc entier et sur un bloc partiel avec ses dispatchs régénérés pour ce nombre de lignes, depuis une sortie mise à zéro : toutes les formes nax sont écartées sur le M3 Ultra, le 4B bf16 revient à ses divergences habituelles aux égalités (35/48 et 28/48 contre la référence bf16, `5-diag/fixed4-bf16/`), le 0.8B à 48/48. **Les mesures herbert sont refaites une seconde fois, au commit `f07e234`** (`logs/1r-bf16-long`, `3r-d32768`, `4r-big`, `5r-ids`, `6r-pieces`, puis `3-d81920` pour tous les moteurs) ; celles du commit `2b86c37` sont gardées sous `*-2b86c37` (une forme nax fausse sur la projection de sortie à certaines tailles de bloc).

Le cadre (2026-10-31, 23 h 30) : sur le Mac Studio seulement, au commit proposé pour le tag public, défauts du moteur (gemm.auto, dec.auto, sdpa.auto), sans plus rien changer au moteur après ces mesures ; six points dans l'ordre, d'une traite, la nuit. Arbre herbert-metal-f16 `9e80538` (sdpa_nax, gemm_nax, `--cool`, `auto.heat` à 0, `autotune_chunk` en médiane), déployé et construit en bf16 (`target/release`) et f16 (`target/f16`) sur m3ultra ; aucun autre moteur LLM n'y tournait (un navigateur à ~175 % CPU sur quatre processus de rendu, noté). mlx-lm 0.32.3, llama.cpp `~/llama.cpp-fb4b2737` (GGUF bf16 et f16 de `$GGUF_DIR`).

## Dossiers

- `logs/1-bf16-long/pp<N>/<modèle>/<moteur>.log` : point 1, herbert bf16 contre llama.cpp bf16 et mlx-lm (bf16), 0.8B, 2B, 4B, 9B à pp2048, 8192, 24576, 61440, et pp81920 pour le 0.8B ; protocole de `2026-10-31-metal-f16-vs-engines` (`_f16-campaign.sh`, 2 runs sans échauffement, tg128 depuis un contexte vide à la suite), l'ordre des moteurs inversé d'une cellule à la suivante (`_art004-1.sh`, sortie `logs/1.out`). Tables : `_tables-1.md` (`_art004_tables.py` ; la ligne herbert est son débit GPU, la table `gpu t/s`).
- `logs/2-f16-long/` : point 2, llama.cpp f16 à pp24576 et pp61440 (mlx-lm y est au point 1, même moteur, mêmes longueurs).
- `logs/3-d32768/`, `logs/3-d81920/` : point 3, tg128 après 32 768 et après 81 920 tokens de contexte, herbert f16 et bf16, llama.cpp bf16 et f16, mlx-lm.
- `logs/4-big/` : point 4, Qwen3.8-27B et Qwen3.6-35B-A3B, pp512, pp8192, tg128, herbert f16 et bf16, llama.cpp bf16, mlx-lm.
- `logs/5-ids/` : point 5, `bench/compare.py` : herbert bf16 contre la référence mlx-lm bf16, herbert f16 contre la référence float16 et contre la bf16, six Qwen, deux prompts, 48 tokens.
- `logs/6-pieces/` : point 6, profils par noyau du 0.8B f16 à pp2048 et pp61440 (`HERBERT_METAL_PROFILE=1`), `--dump-msl` des noyaux émis au chargement.
- `logs/rest.out` : la sortie du pilote des points 2 à 6 (`_art004-rest.sh`).

## Point 1 (bf16 aux prompts longs, tokens/s, régime de `2026-10-31-metal-f16-vs-engines`) — herbert bf16 / mlx-lm / llama.cpp bf16

| modèle | pp2048 | pp8192 | pp24576 | pp61440 |
| --- | --- | --- | --- | --- |
| 0.8B | 13 375 / 10 517 / 9 377 (×1,27) | 12 264 / 10 109 / 8 519 (×1,21) | 9 263 / 8 478 / 6 308 (×1,09) | 6 096 / 5 692 / 3 926 (×1,07) |
| 2B | 6 092 / 5 541 / 4 723 (×1,10) | 5 905 / 5 433 / 4 492 (×1,09) | 5 111 / 4 829 / 3 784 (×1,06) | 3 978 / 3 816 / 2 773 (×1,04) |
| 4B | 2 465 / 2 137 / 1 897 (×1,15) | 2 377 / 2 087 / 1 782 (×1,14) | 2 050 / 1 840 / 1 507 (×1,11) | 1 573 / 1 428 / 1 113 (×1,10) |
| 9B | 1 351 / 1 209 / 1 115 (×1,12) | 1 326 / 1 200 / 1 075 (×1,11) | 1 217 / 1 118 / 969 (×1,09) | 1 032 / 954 / 790 (×1,08) |

Herbert bf16 devant mlx-lm et llama.cpp aux seize cellules, de ×1,27 à ×1,04 ; l'avance se resserre avec la longueur. **0.8B pp81920 : 4 966 / 4 784 / 3 232 (×1,04)** là où `2026-10-31-metal-f16-vs-engines` (avant `sdpa_mma`) avait 1 128 / 4 848 / 3 240 : ×4,4 sur herbert, de ×0,23 à ×1,04 de mlx-lm. Le point 1 a fini à 00 h 36 (64 min).

## Point 2 (llama.cpp f16 à pp24576 et pp61440, 2 runs ; mlx-lm au point 1) — fini à 00 h 41

| modèle | pp24576 | pp61440 |
| --- | ---: | ---: |
| 0.8B | 6 316 | 3 929 |
| 2B | 3 799 | 2 781 |
| 4B | 1 546 | 1 128 |
| 9B | (GGUF F16 absent : converti et mesuré à la fin, `_art004-9bf16.sh`) | |

Le f16 de llama.cpp vaut son bf16 à ±1 % (point 1 : 6 308 / 3 926, 3 784 / 2 773, 1 507 / 1 113). Les valeurs approchées de `2026-10-31-metal-f16-sdpa-long` : à pp61440, herbert f16 0.8B 5 140 contre llama.cpp f16 3 929 et mlx-lm 5 692 ; à relire avec herbert f16 au point 3 et au commit du tag (le point 1 est en bf16 : 6 096).

## Point 3 (tg128 après 32 768 tokens de contexte, 2 runs ; herbert f16 / herbert bf16 / mlx-lm / llama.cpp bf16 / llama.cpp f16) — fini à 01 h 11

| modèle | herbert f16 | herbert bf16 | mlx-lm | llama.cpp bf16 | llama.cpp f16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.8B | 296 | 289 | 201 | 174 | 177 |
| 2B | 161 | 157 | 125 | 121 | 122 |
| 4B | 69 | 69 | 62 | 59 | 60 |
| 9B | 43 | 42 | 38 | 38 | (GGUF F16 : fin de campagne) |

Devant partout en contexte long sur le M3 Ultra (×1,47, ×1,29, ×1,11, ×1,13 sur mlx-lm) : pas le cas du M1 de `2026-10-31-metal-f16-vs-engines` (15,3 contre 21,5 après 81 920). `_tables-3-d32768.md`. Le tg128 après 81 920 tokens tourne en dernier (`3-d81920/`).

## Point 4 (Qwen3.8-27B et Qwen3.6-35B-A3B : pp512 / pp8192 / tg128, 2 runs après un échauffement) — fini à 01 h 50

| modèle | herbert f16 | herbert bf16 | llama.cpp bf16 | mlx-lm |
| --- | --- | --- | --- | --- |
| 27B | 380 / 386 / 14,8 | 379 / 389 / 14,8 | 320 / 307 / 13,5 | échec au chargement (GPU Timeout dans `mx.eval` des paramètres, 01 h 33, juste après deux chargements de 54 Go par herbert) : second essai en fin de campagne (`_art004-mlx27b.sh`, `mlx-failed-1.log`) |
| 35B-A3B | 2 448 / 3 675 / 90,8 | 2 433 / 3 634 / 89,2 | 2 004 / 1 804 / 71,7 | 1 893 / 2 487 / 73,3–78,4 |

Herbert devant sur les deux (27B : ×1,19–1,26 sur llama.cpp au prefill, ×1,10 au décodage ; 35B-A3B : ×1,22–1,48 sur llama.cpp, ×1,29–1,48 sur mlx-lm au prefill, ×1,16–1,24 au décodage). Sur le 27B, `sdpa_mma` ne se construit pas (24 têtes : 38 400 octets de mémoire de groupe, maximum 32 768) et `sdpa.auto` prend `sdpa_nax` (0,81 ms contre `sdpa_mm2d` 1,41 à 512 lignes : le noyau à fragments gagne aussi sur le M3 Ultra à cette forme) ; le réglage au chargement du 27B dure 384 s.

## Ajouts

- `logs/2-f16-long/Qwen3.5-9B/`, `3-d32768/Qwen3.5-9B/llama-f16.log`, `3-d81920/Qwen3.5-9B/llama-f16.log` : le 9B F16 (`~/gguf-f16/Qwen3.5-9B-F16.gguf`, 17,9 Go, `llama-quantize … F16` depuis le BF16, `9B-f16-quantize.log`, `_art004-9bf16.sh`).
- mlx-lm sur le 27B bf16 : échec au chargement deux fois (« GPU Timeout Error » dans `mx.eval(model.parameters())`, 01 h 33 après herbert, 02 h 23 seul sur la machine : `4-big/Qwen3.8-27B/mlx-failed-1.log`, `mlx.log`) : mlx-lm 0.32.3 ne charge pas Qwen3.8-27B en bf16 sur ce M3 Ultra.

## Point 1 refait au commit `2b86c37` (`logs/1r-bf16-long`, 02 h 28 → 02 h 53) — herbert bf16 contre mlx-lm et llama.cpp bf16 du premier passage

| modèle | longueur | herbert bf16 (2b86c37) | mlx-lm | llama.cpp bf16 | h/mlx |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.8B | pp2048 | 12 302 | 10 517 | 9 377 | ×1.17 |
| 2B | pp2048 | 5 626 | 5 541 | 4 723 | ×1.02 |
| 4B | pp2048 | 2 281 | 2 137 | 1 897 | ×1.07 |
| 9B | pp2048 | 1 261 | 1 209 | 1 115 | ×1.04 |
| 0.8B | pp8192 | 11 292 | 10 109 | 8 519 | ×1.12 |
| 2B | pp8192 | 5 478 | 5 433 | 4 492 | ×1.01 |
| 4B | pp8192 | 2 204 | 2 087 | 1 782 | ×1.06 |
| 9B | pp8192 | 1 236 | 1 200 | 1 075 | ×1.03 |
| 0.8B | pp24576 | 8 685 | 8 478 | 6 308 | ×1.02 |
| 2B | pp24576 | 4 782 | 4 829 | 3 784 | ×0.99 |
| 4B | pp24576 | 1 909 | 1 840 | 1 507 | ×1.04 |
| 9B | pp24576 | 1 136 | 1 118 | 969 | ×1.02 |
| 0.8B | pp61440 | 5 643 | 5 692 | 3 926 | ×0.99 |
| 2B | pp61440 | 3 689 | 3 816 | 2 773 | ×0.97 |
| 4B | pp61440 | 1 456 | 1 428 | 1 113 | ×1.02 |
| 9B | pp61440 | 961 | 954 | 790 | ×1.01 |
| 0.8B | pp81920 | 4 756 | 4 784 | 3 232 | ×0.99 |

Devant de ×1,17 à ×1,02 jusqu'à pp8192 ; parité de pp24576 à pp81920 (×0,97 à ×1,04), le 2B un à trois points derrière mlx-lm aux deux longueurs longues. Au commit faux (`9e80538`) les mêmes cellules donnaient ×1,27 → ×1,04 : la part des noyaux faux. `_tables-1r.md`, `_tables-1r-vs.md`.

## Point 3 refait au commit `2b86c37` (`logs/3r-d32768`) — tg128 après 32 768 tokens, herbert f16 / herbert bf16, contre mlx-lm et llama.cpp du premier passage

| modèle | herbert f16 | herbert bf16 | mlx-lm | llama.cpp bf16 | llama.cpp f16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.8B | 297 | 297 | 201 | 174 | 177 |
| 2B | 160 | 160 | 125 | 121 | 122 |
| 4B | 72 | 71 | 62 | 59 | 60 |
| 9B | 43 | 43 | 38 | 38 | (fin de campagne, `3-d32768/Qwen3.5-9B/llama-f16.log`) |

Le décodage ne passe pas par les noyaux nax (matvec, `sdpa_decode`) : les débits sont ceux du premier passage, justes cette fois pour le contexte qui les précède. Devant partout, ×1,13–1,48 sur mlx-lm. `_tables-3r-d32768.md`.

## Point 4 refait au commit `2b86c37` (`logs/4r-big`) — pp512 / pp8192 / tg128, contre llama.cpp bf16 et mlx-lm du premier passage

| modèle | herbert f16 | herbert bf16 | llama.cpp bf16 | mlx-lm |
| --- | --- | --- | --- | --- |
| 27B | 353 / 351 / 14,8 | 353 / 351 / 14,8 | 320 / 307 / 13,5 | ne charge pas |
| 35B-A3B | 2 364 / 3 208 / 91,3 | 2 347 / 3 191 / 90,0 | 2 004 / 1 804 / 71,7 | 1 893 / 2 487 / 73,3–78,4 |

Devant : 27B ×1,10–1,14 sur llama.cpp au prefill, ×1,10 au décodage ; 35B-A3B ×1,18–1,78 sur llama.cpp, ×1,25–1,29 sur mlx-lm au prefill, ×1,16–1,25 au décodage. Sur le 27B l'attention retenue est `sdpa_mm2d` (`sdpa_mma` ne se construit pas pour 24 têtes, `sdpa_nax` est écarté par le contrôle).

## Point 1 refait au commit `f07e234` (`logs/1r-bf16-long`, 04 h 08 → 04 h 33 ; **les chiffres de l'article**) — herbert bf16 contre mlx-lm et llama.cpp bf16 du premier passage

| modèle | longueur | herbert bf16 (f07e234) | mlx-lm | llama.cpp bf16 | h/mlx |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.8B | pp2048 | 12 775 | 10 517 | 9 377 | ×1.21 |
| 2B | pp2048 | 5 884 | 5 541 | 4 723 | ×1.06 |
| 4B | pp2048 | 2 386 | 2 137 | 1 897 | ×1.12 |
| 9B | pp2048 | 1 315 | 1 209 | 1 115 | ×1.09 |
| 0.8B | pp8192 | 11 726 | 10 109 | 8 519 | ×1.16 |
| 2B | pp8192 | 5 676 | 5 433 | 4 492 | ×1.04 |
| 4B | pp8192 | 2 289 | 2 087 | 1 782 | ×1.10 |
| 9B | pp8192 | 1 286 | 1 200 | 1 075 | ×1.07 |
| 0.8B | pp24576 | 8 927 | 8 478 | 6 308 | ×1.05 |
| 2B | pp24576 | 4 932 | 4 829 | 3 784 | ×1.02 |
| 4B | pp24576 | 1 971 | 1 840 | 1 507 | ×1.07 |
| 9B | pp24576 | 1 178 | 1 118 | 969 | ×1.05 |
| 0.8B | pp61440 | 5 751 | 5 692 | 3 926 | ×1.01 |
| 2B | pp61440 | 3 771 | 3 816 | 2 773 | ×0.99 |
| 4B | pp61440 | 1 493 | 1 428 | 1 113 | ×1.05 |
| 9B | pp61440 | 989 | 954 | 790 | ×1.04 |
| 0.8B | pp81920 | 4 827 | 4 784 | 3 232 | ×1.01 |

Devant à toutes les cellules sauf le 2B à pp61440 (×0,99) : ×1,21 → ×1,01 sur le 0.8B de pp2048 à pp81920, ×1,06 → ×0,99 sur le 2B, ×1,12 → ×1,05 sur le 4B, ×1,09 → ×1,04 sur le 9B ; llama.cpp bf16 à ×0,65–0,80 de herbert. Aucune forme nax ni SwiGLU fusionné retenue (le contrôle les écarte toutes sur le M3 Ultra). `_tables-1r.md`, `_tables-1r-vs.md`.

## Point 3 refait au commit `f07e234` (`logs/3r-d32768` ; **les chiffres de l'article**) — tg128 après 32 768 tokens

| modèle | herbert f16 | herbert bf16 | mlx-lm | llama.cpp bf16 | llama.cpp f16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.8B | 298 | 298 | 201 | 174 | 177 |
| 2B | 155 | 154 | 125 | 121 | 122 |
| 4B | 72 | 69 | 62 | 59 | 60 |
| 9B | 42 | 43 | 38 | 38 | (`3-d32768/Qwen3.5-9B/llama-f16.log`) |

Devant partout, ×1,11–1,48 sur mlx-lm. La version `2b86c37` (`3r-d32768-2b86c37`) donnait 297 / 160 / 72 / 43 : le décodage ne passe pas par les noyaux nax.

## Point 4 refait au commit `f07e234` (`logs/4r-big` ; **les chiffres de l'article**) — pp512 / pp8192 / tg128

| modèle | herbert f16 | herbert bf16 | llama.cpp bf16 | mlx-lm |
| --- | --- | --- | --- | --- |
| 27B | 374 / 365 / 14,9 | 374 / 365 / 14,8 | 320 / 307 / 13,5 | ne charge pas (GPU Timeout, deux fois) |
| 35B-A3B | 2 430 / 3 267 / 91,0 | 2 415 / 3 248 / 90,9 | 2 004 / 1 804 / 71,7 | 1 893 / 2 487 / 73,3–78,4 |

27B ×1,17–1,19 sur llama.cpp au prefill, ×1,10 au décodage ; 35B-A3B ×1,21–1,81 sur llama.cpp et ×1,28–1,31 sur mlx-lm au prefill, ×1,16–1,27 au décodage.

## Point 5 au commit `f07e234` (`logs/5r-ids`, `bench/compare.py`, six Qwen, deux prompts, 48 tokens ; **les chiffres de l'article**)

Résumé : herbert f16 contre la référence mlx-lm float16 **12/12 runs identiques** (576/576 tokens) ; herbert bf16 contre la référence bf16 7/12 identiques, herbert f16 contre la référence bf16 7/12. Les 10 divergences se classent toutes : 8 égalités exactes de la référence (deux tokens au même logit, à 0,06 ou 0,125 de résolution bf16 : l'ordre est tranché par l'arrondi) et 2 écarts d'un ULP bf16 (la référence sépare ses deux premiers de 0,125, nous les mettons à égalité ou dans l'autre ordre) ; après la divergence, les tokens suivants restent largement communs (le 2B : 321, 1428, 6165, 4779). Aucune divergence de calcul. `_tables-5r.md`, les sorties complètes dans `5r-ids/*.out`, les logs des chats dans `5r-ids/<paire>/`.

| paire | modèle | prompt | identiques | nature | première différence |
| --- | --- | --- | ---: | --- | --- |
| herbert bf16 / réf. bf16 | 0.8B | short | 48/48 | — | — |
| herbert bf16 / réf. bf16 | 0.8B | long | 48/48 | — | — |
| herbert bf16 / réf. bf16 | 2B | short | 48/48 | — | — |
| herbert bf16 / réf. bf16 | 2B | long | 25/48 | égalité exacte de la référence | token 25 : réf. 13263 22.5 / 3712 22.5 (marge 0) ; herbert 3712 22.62 / 13263 22.5 (marge 0.125) |
| herbert bf16 / réf. bf16 | 4B | short | 48/48 | — | — |
| herbert bf16 / réf. bf16 | 4B | long | 28/48 | égalité exacte de la référence | token 28 : réf. 14791 21.88 / 1141 21.88 (marge 0) ; herbert 1141 21.88 / 14791 21.88 (marge 0) |
| herbert bf16 / réf. bf16 | 9B | short | 48/48 | — | — |
| herbert bf16 / réf. bf16 | 9B | long | 48/48 | — | — |
| herbert bf16 / réf. bf16 | 3.8-27B | short | 15/48 | écart d'un ULP (réf. marge 0.125) | token 15 : réf. 318 27 / 303 26.88 (marge 0.125) ; herbert 303 27 / 318 27 (marge 0) |
| herbert bf16 / réf. bf16 | 3.8-27B | long | 13/48 | égalité exacte de la référence | token 13 : réf. 6558 32.5 / 561 32.5 (marge 0) ; herbert 561 32.75 / 6558 32.75 (marge 0) |
| herbert bf16 / réf. bf16 | 3.6-35B-A3B | short | 48/48 | — | — |
| herbert bf16 / réf. bf16 | 3.6-35B-A3B | long | 24/48 | égalité exacte de la référence | token 24 : réf. 26623 27.75 / 8482 27.75 (marge 0) ; herbert 8482 27.75 / 26623 27.62 (marge 0.125) |
| herbert f16 / réf. float16 | 0.8B | short | 48/48 | — | — |
| herbert f16 / réf. float16 | 0.8B | long | 48/48 | — | — |
| herbert f16 / réf. float16 | 2B | short | 48/48 | — | — |
| herbert f16 / réf. float16 | 2B | long | 20/48 | égalité exacte de la référence | token 20 : réf. 2716 24.78 / 1414 24.78 (marge 0) ; herbert 1414 24.77 / 2716 24.77 (marge 0) |
| herbert f16 / réf. float16 | 4B | short | 48/48 | — | — |
| herbert f16 / réf. float16 | 4B | long | 48/48 | — | — |
| herbert f16 / réf. float16 | 9B | short | 48/48 | — | — |
| herbert f16 / réf. float16 | 9B | long | 48/48 | — | — |
| herbert f16 / réf. float16 | 3.8-27B | short | 48/48 | — | — |
| herbert f16 / réf. float16 | 3.8-27B | long | 48/48 | — | — |
| herbert f16 / réf. float16 | 3.6-35B-A3B | short | 48/48 | — | — |
| herbert f16 / réf. float16 | 3.6-35B-A3B | long | 48/48 | — | — |
| herbert f16 / réf. bf16 | 0.8B | short | 48/48 | — | — |
| herbert f16 / réf. bf16 | 0.8B | long | 43/48 | écart d'un ULP (réf. marge 0.125) | token 43 : réf. 3992 16.75 / 4779 16.62 (marge 0.125) ; herbert 4779 16.72 / 3992 16.7 (marge 0.01562) |
| herbert f16 / réf. bf16 | 2B | short | 48/48 | — | — |
| herbert f16 / réf. bf16 | 2B | long | 25/48 | égalité exacte de la référence | token 25 : réf. 13263 22.5 / 3712 22.5 (marge 0) ; herbert 3712 22.52 / 13263 22.5 (marge 0.01562) |
| herbert f16 / réf. bf16 | 4B | short | 48/48 | — | — |
| herbert f16 / réf. bf16 | 4B | long | 48/48 | — | — |
| herbert f16 / réf. bf16 | 9B | short | 48/48 | — | — |
| herbert f16 / réf. bf16 | 9B | long | 48/48 | — | — |
| herbert f16 / réf. bf16 | 3.8-27B | short | 20/48 | égalité exacte de la référence | token 20 : réf. 45690 26.75 / 907 26.75 (marge 0) ; herbert 907 26.78 / 45690 26.67 (marge 0.1094) |
| herbert f16 / réf. bf16 | 3.8-27B | long | 13/48 | égalité exacte de la référence | token 13 : réf. 6558 32.5 / 561 32.5 (marge 0) ; herbert 561 32.66 / 6558 32.47 (marge 0.1875) |
| herbert f16 / réf. bf16 | 3.6-35B-A3B | short | 48/48 | — | — |
| herbert f16 / réf. bf16 | 3.6-35B-A3B | long | 24/48 | égalité exacte de la référence | token 24 : réf. 26623 27.75 / 8482 27.75 (marge 0) ; herbert 8482 27.77 / 26623 27.69 (marge 0.07812) |

## Point 6 au commit `f07e234` (`logs/6r-pieces`) — les pièces de l'article

- **La part de l'attention dans le prefill après `sdpa_mma`** (0.8B f16, M3 Ultra, `HERBERT_METAL_PROFILE=1`, défauts du moteur) : **6,4 % à pp2048** (`sdpa_mma_h8kv2d256_q16k32_st_qr` 77 ms sur 1 207 ms de GPU comptés ; avant : 21 %) et **41,3 % à pp61440** (8 113 ms sur 19 646 ms ; avant : 89,5 %). Réserve : ce profil compte aussi les dispatchs du réglage au chargement (les candidats chronométrés), les parts sont donc approchées ; un profil à noyaux figés identiques à ceux choisis (`gemm.mma=2,gemm.tn=32,sdpa.prefill=3,sdpa.qreg=1`), un seul prefill compté, suit en fin de campagne (`profile-fixed-0.8B-pp*.log`, `_art004-prof.sh`). Les débits de ces runs : pp2048 12 971 tokens/s GPU, pp61440 5 718.
- **Une ligne `[autotune]` avec le temps de mesure au chargement** (`dump-msl.log`, 0.8B f16, pp512) : `[autotune] sdpa 512 rows at 512: flash 1.444 ms, mma 0.533 ms, mm2d 0.825 ms, nax* 0.461 ms -> mma` (l'étoile : écarté par le contrôle, « wrong against flash, never picked: nax (512 rows: max err 7.790e-2 of 9.082e-2) ») et `[metal] 688 kernels compiled in 0.11 s, GEMMs and blocks timed in 24.11 s, 137 dispatches per token, blocks of 2048 rows`. Pour le 0.8B à pp2048 : `GEMMs and blocks timed in 19.99 s` ; à pp61440 : `35.05 s` (`profile-0.8B-pp*.log`).
- **L'extrait `--dump-msl` du noyau `sdpa_mma`** : `_sdpa_mma_h8kv2d256_q16k32_st_qr.metal` (copie de `logs/6r-pieces/msl/sdpa_mma_h8kv2d256_q16k32_st_qr.metal`, le MSL émis par Rust : 8 têtes, 2 têtes K/V, dimension 256, 16 requêtes × 32 clés par tuile, K/V étagés par le groupe, Q en registres) ; les 688 noyaux émis au chargement sont dans `msl/`.

## Point 3 bis au commit `f07e234` (`logs/3-d81920`, 06 h 23 → 07 h 11) — tg128 après 81 920 tokens, 1 run après un échauffement, cinq moteurs, ordre inversé d'un modèle au suivant

| modèle | herbert f16 | herbert bf16 | mlx-lm | llama.cpp bf16 | llama.cpp f16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.8B | 224 | 199 | 156 | 154 | 156 |
| 2B | 134 | 136 | 105 | 109 | 110 |
| 4B | 57 | 57 | 52 | 47 | 47 |
| 9B | 35 | 37 | 34 | 33 | 33 |

Devant partout : ×1,28–1,44 (0.8B), ×1,28–1,29 (2B), ×1,10 (4B), ×1,03–1,09 (9B) sur mlx-lm. Le M1 de `2026-10-31-metal-f16-vs-engines` était derrière après 81 920 (15,3 contre 21,5) ; le M3 Ultra ne l'est pas. `_tables-3b-d81920.md`. **La campagne a fini à 07 h 11** ; `profile-fixed-0.8B-pp*.log` (point 6, profils à noyaux figés) suit.

## Point 6, complément : les profils à noyaux figés (`logs/6r-pieces/profile-fixed-0.8B-pp*.log`, 07 h 11 ; les noyaux que les défauts ont choisis, `gemm.mma=2,gemm.tn=32,sdpa.prefill=3,sdpa.qreg=1`, un seul prefill compté)

- **La part de l'attention `sdpa_mma` dans le prefill du 0.8B f16 : 6.1 % (9.60 ms sur 158.0 ms) à pp2048, 59,6 % à pp61440 (6 360 ms sur 10 676 ms)** ; avant `sdpa_mma` (`sdpa_flash`, `2026-10-31-metal-f16-sdpa-long`) : 21 % et 89,5 %. Ce sont les chiffres à publier ; ceux du profil par défaut (6,4 % et 41,3 %) comptaient les dispatchs du réglage au chargement.
- Débits de ces runs : pp2048 12 966 tokens/s GPU, pp61440 5 755 (identiques au profil par défaut : les mêmes noyaux).

