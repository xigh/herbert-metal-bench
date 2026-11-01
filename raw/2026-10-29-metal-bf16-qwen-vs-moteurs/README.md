# 2026-10-29 — herbert-metal-bf16 (`gemm.auto=1`) contre llama.cpp, mlx-lm, Rapid-MLX et BaseRT, six Qwen bf16 sur m3ultra, deux tours

Le protocole de la campagne du 2026-10-28 (non publiée ici), refait après le travail des jours précédents sur le prefill (`matmul2d`, choix au chargement, blocs de 2 048 lignes, tri des experts). Logs bruts ; outillage préfixé `_`.

Code : herbert-metal-bf16 `9e3c939` (`484800c` ne change que la doc) ; binaires m3ultra sha256 `72988aab…` (herbert-metal-bench), `cf3d0f88…` (herbert-metal-chat). Réglages par défaut : la mesure au chargement (`gemm.auto=1`) choisit noyaux, tuiles, tri et blocs ; ses choix sont dans chaque `herbert.log` (`[autotune]`). Les autres moteurs comme le 2026-10-28 : llama.cpp-fb4b2737 fb4b2737 (GGUF bf16), mlx-lm 0.32.0 / mlx 0.32.3, Rapid-MLX 0.16.10, BaseRT 0.4.2.

Protocole (`_vs.sh`, sortie complète `_vs.out`, 2026-10-28 23:42 → 2026-10-29 00:55) : ids contre mlx-lm (`campagne/ids*`), puis deux tours de `_qwen-campaign.sh` (pp512 sur ids pseudo-aléatoires depuis un contexte vide, tg128, un échauffement, 3 runs, 30 s entre deux moteurs) ; l'ordre des moteurs tourne d'un modèle à l'autre comme le 2026-10-28 et le tour B le prend à l'envers. BaseRT ni sur le 27B (pagination) ni sur le 35B (ne tient pas) ; ses `.base` du 2B, 4B et 9B convertis au tour A, supprimés après le tour B. Aucun autre moteur ne tournait avant ni après. Tableaux : `_campaign_table.py campagne/A|B` → `table-A.md`, `table-B.md` (moyenne ± écart-type des 3 runs).

Incident : au tour A, mlx-lm sur le 27B a échoué au chargement (`[METAL] Command buffer execution failed: Caused GPU Timeout Error` dans `mx.eval(model.parameters())`, `campagne/A/Qwen3.8-27B/mlx.log`) ; au tour B il passe.

Ids (`gemm.auto=1`) : identiques à ceux des mesures des jours précédents sur les six Qwen (`campagne/ids.txt`).

## Tour A
pp512 (tok/s)

| model | herbert-metal-bf16 | llama.cpp | mlx-lm | Rapid-MLX | BaseRT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | **11 312.2 ± 8.2** | 9 940.8 ± 11.8 | 8 866.3 ± 85.1 | 7 284.3 ± 76.6 | 9 412.1 ± 11.7 |
| Qwen3.5-2B | **5 368.5 ± 4.4** | 4 829.0 ± 6.2 | 4 442.2 ± 11.9 | 3 654.1 ± 236.2 | 4 534.5 ± 5.5 |
| Qwen3.5-4B | **2 223.0 ± 1.1** | 1 928.5 ± 1.6 | 1 794.1 ± 0.6 | 1 653.9 ± 26.4 | 1 869.2 ± 0.9 |
| Qwen3.5-9B | **1 245.4 ± 0.9** | 1 124.8 ± 0.3 | 1 009.6 ± 0.2 | 953.4 ± 15.5 | 1 043.3 ± 0.1 |
| Qwen3.8-27B | **370.7 ± 0.2** | 319.9 ± 0.1 | — | 270.6 ± 4.6 | — |
| Qwen3.6-35B-A3B | **2 355.5 ± 1.1** | 2 010.3 ± 8.6 | 1 903.7 ± 2.8 | 1 761.8 ± 26.6 | — |

tg128 (tok/s)

| model | herbert-metal-bf16 | llama.cpp | mlx-lm | Rapid-MLX | BaseRT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 238.2 ± 0.2 | 215.2 ± 2.7 | 251.0 ± 0.3 | 254.5 ± 0.1 | **307.5 ± 0.8** |
| Qwen3.5-2B | 135.9 ± 0.1 | 135.2 ± 1.6 | 143.9 ± 0.0 | 144.9 ± 0.1 | **161.4 ± 0.2** |
| Qwen3.5-4B | 69.7 ± 0.0 | 69.1 ± 0.0 | 71.4 ± 0.0 | 71.2 ± 0.0 | **77.4 ± 0.0** |
| Qwen3.5-9B | 40.8 ± 0.0 | 41.4 ± 0.1 | 41.0 ± 0.0 | 40.9 ± 0.0 | **43.8 ± 0.0** |
| Qwen3.8-27B | 13.2 ± 0.0 | **13.5 ± 0.0** | — | 13.3 ± 0.0 | — |
| Qwen3.6-35B-A3B | 70.6 ± 0.0 | 71.8 ± 0.0 | 73.8 ± 0.1 | **78.2 ± 0.0** | — |

## Tour B
pp512 (tok/s)

| model | herbert-metal-bf16 | llama.cpp | mlx-lm | Rapid-MLX | BaseRT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | **11 203.0 ± 9.3** | 9 850.4 ± 54.3 | 8 913.7 ± 135.4 | 7 330.0 ± 90.7 | 9 412.6 ± 16.7 |
| Qwen3.5-2B | **5 359.6 ± 5.6** | 4 815.4 ± 15.4 | 4 463.4 ± 18.6 | 3 554.8 ± 257.5 | 4 532.1 ± 3.9 |
| Qwen3.5-4B | **2 220.3 ± 0.6** | 1 924.0 ± 5.0 | 1 796.5 ± 0.8 | 1 656.9 ± 26.7 | 1 866.2 ± 0.6 |
| Qwen3.5-9B | **1 243.5 ± 1.0** | 1 123.5 ± 0.2 | 1 009.2 ± 1.3 | 953.2 ± 16.5 | 1 043.3 ± 0.5 |
| Qwen3.8-27B | **369.8 ± 0.2** | 319.6 ± 0.2 | 286.2 ± 0.0 | 270.9 ± 4.5 | — |
| Qwen3.6-35B-A3B | **2 353.3 ± 1.2** | 2 005.7 ± 8.3 | 1 884.3 ± 5.8 | 1 752.4 ± 18.4 | — |

tg128 (tok/s)

| model | herbert-metal-bf16 | llama.cpp | mlx-lm | Rapid-MLX | BaseRT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 235.5 ± 0.2 | 214.0 ± 3.7 | 250.3 ± 0.2 | 253.6 ± 0.3 | **304.2 ± 0.9** |
| Qwen3.5-2B | 134.8 ± 0.1 | 134.6 ± 0.5 | 143.7 ± 0.0 | 144.3 ± 0.1 | **160.5 ± 0.3** |
| Qwen3.5-4B | 69.5 ± 0.0 | 68.6 ± 0.2 | 71.4 ± 0.0 | 71.3 ± 0.0 | **77.3 ± 0.0** |
| Qwen3.5-9B | 40.8 ± 0.0 | 41.1 ± 0.1 | 41.1 ± 0.0 | 40.9 ± 0.0 | **43.8 ± 0.0** |
| Qwen3.8-27B | 13.1 ± 0.0 | 13.4 ± 0.0 | **13.4 ± 0.0** | 13.3 ± 0.0 | — |
| Qwen3.6-35B-A3B | 70.0 ± 0.0 | 70.0 ± 0.2 | 73.4 ± 0.0 | **77.3 ± 0.1** | — |
