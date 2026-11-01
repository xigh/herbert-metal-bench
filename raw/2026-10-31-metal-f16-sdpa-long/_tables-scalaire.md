## M3 Ultra — logs/m3ultra

| méthode | pp2048 | pp8192 | pp24576 | pp61440 |
| --- | ---: | ---: | ---: | ---: |
| base | 10 878 | 6 707 | 3 323 | 1 467 |
| blk512 | 9 665 ×0.89 | 5 877 ×0.88 | 2 812 ×0.85 | 1 265 ×0.86 |
| blk8192 | 10 901 ×1.00 | 7 241 ×1.08 | 3 660 ×1.10 | 1 701 ×1.16 |
| q8k32 | 10 706 ×0.98 | 6 068 ×0.90 | 2 717 ×0.82 | 1 096 ×0.75 |
| rows | 9 865 ×0.91 | 4 439 ×0.66 | 1 657 ×0.50 |  |
| simt | 5 139 ×0.47 | 3 969 ×0.59 | 2 482 ×0.75 | 1 277 ×0.87 |

Profil : 27 dispatchs d'attention à pp2048 pour 6 couches, 21 de la mesure au chargement ; 0.778 du profil de pp2048 retiré, noyau par noyau, de chaque longueur.

| profil (base) | pp2048 | pp8192 | pp24576 | pp61440 |
| --- | ---: | ---: | ---: | ---: |
| attention (sdpa_*) | 21.1 % | 51.9 % | 76.1 % | 89.5 % |
| GEMM (gemm_*) | 57.8 % | 36.1 % | 17.7 % | 7.7 % |
| DeltaNet (dn_*) | 16.4 % | 9.4 % | 4.9 % | 2.2 % |
| reste | 4.7 % | 2.6 % | 1.3 % | 0.6 % |
| GPU total (s) | 0.11 | 1.17 | 7.30 | 42.03 |
| attention (s) | 0.023 | 0.606 | 5.558 | 37.630 |
| attention, TFLOPS | 4.54 | 2.72 | 2.67 | 2.47 |

pp2048 : sdpa_flash_h8kv2d256_q16k16 21.1 % ; gemm_mm2d_1024x7168_bf16_t32_mf 19.6 % ; gemm_mm2d_1024x8224_bf16_t32_mf 13.3 % ; dn_mma_state_k16v16_c32d32 8.9 %
pp8192 : sdpa_flash_h8kv2d256_q16k16 51.9 % ; gemm_mm2d_1024x7168_bf16_t32_mf 11.4 % ; gemm_mm2d_1024x8224_bf16_t32_mf 9.4 % ; gemm_mm2d_3584x1024_res_t64_r32_mf 8.0 %
pp24576 : sdpa_flash_h8kv2d256_q16k16 76.1 % ; gemm_mm2d_1024x7168_bf16_t32_mf 5.9 % ; gemm_mm2d_1024x8224_bf16_t32_mf 5.2 % ; gemm_mm2d_3584x1024_res_t64_r32_mf 3.5 %
pp61440 : sdpa_flash_h8kv2d256_q16k16 89.5 % ; gemm_mm2d_1024x7168_bf16_t32_mf 2.6 % ; gemm_mm2d_1024x8224_bf16_t32_mf 2.2 % ; gemm_mm2d_3584x1024_res_t64_r32_mf 1.5 %

## M1 Air — logs/m1air

| méthode | pp2048 | pp8192 | pp24576 | pp61440 |
| --- | ---: | ---: | ---: | ---: |
| base | 1 172 | 609 | 264 | 115 |
| blk512 | 1 160 ×0.99 | 598 ×0.98 | 258 ×0.98 | 112 ×0.98 |
| blk8192 | 1 164 ×0.99 | 605 ×0.99 | 258 ×0.98 | 112 ×0.97 |
| q8k32 | 1 160 ×0.99 | 549 ×0.90 | 228 ×0.86 | 99 ×0.86 |
| rows | 1 047 ×0.89 | 420 ×0.69 |  |  |
| simt | 217 ×0.19 | 186 ×0.31 | 132 ×0.50 | 81 ×0.70 |

Profil : 33 dispatchs d'attention à pp2048 pour 6 couches, 27 de la mesure au chargement ; 0.818 du profil de pp2048 retiré, noyau par noyau, de chaque longueur.

| profil (base) | pp2048 | pp8192 | pp24576 | pp61440 |
| --- | ---: | ---: | ---: | ---: |
| attention (sdpa_*) | 31.3 % | 67.0 % | 85.4 % | 93.6 % |
| GEMM (gemm_*) | 56.5 % | 27.3 % | 12.1 % | 5.3 % |
| DeltaNet (dn_*) | 10.4 % | 4.9 % | 2.2 % | 1.0 % |
| reste | 1.8 % | 0.8 % | 0.3 % | 0.1 % |
| GPU total (s) | 0.79 | 12.49 | 93.88 | 546.66 |
| attention (s) | 0.249 | 8.364 | 80.150 | 511.461 |
| attention, TFLOPS | 0.41 | 0.20 | 0.19 | 0.18 |

pp2048 : sdpa_flash_h8kv2d256_q16k16 31.3 % ; gemm_mm2d_1024x7168_bf16_t64_mf 19.8 % ; gemm_mm2d_1024x8224_bf16_t64_mf 13.8 % ; gemm_mm2d_3584x1024_res_t32_mf 8.1 %
pp8192 : sdpa_flash_h8kv2d256_q16k16 67.0 % ; gemm_mm2d_1024x7168_bf16_t64_mf 9.6 % ; gemm_mm2d_1024x8224_bf16_t64_mf 8.1 % ; gemm_mm2d_3584x1024_res_t32_mf 4.8 %
pp24576 : sdpa_flash_h8kv2d256_q16k16 85.4 % ; gemm_mm2d_1024x7168_bf16_t64_mf 4.2 % ; gemm_mm2d_1024x8224_bf16_t64_mf 3.6 % ; gemm_mm2d_3584x1024_res_t32_mf 2.2 %
pp61440 : sdpa_flash_h8kv2d256_q16k16 93.6 % ; gemm_mm2d_1024x7168_bf16_t64_mf 1.9 % ; gemm_mm2d_1024x8224_bf16_t64_mf 1.6 % ; gemm_mm2d_3584x1024_res_t32_mf 1.0 %

## M5 — logs/m5air

| méthode | pp2048 | pp8192 | pp24576 | pp61440 |
| --- | ---: | ---: | ---: | ---: |
| base | 5 002 | 2 197 | 804 | 299 |
| blk512 | 4 955 ×0.99 | 2 215 ×1.01 | 806 ×1.00 | 307 ×1.03 |
| blk8192 | 4 552 ×0.91 | 1 823 ×0.83 | 772 ×0.96 | 300 ×1.00 |
| q8k32 | 4 888 ×0.98 | 2 003 ×0.91 | 705 ×0.88 | 238 ×0.80 |
| rows | 3 801 ×0.76 | 1 204 ×0.55 |  |  |
| simt | 1 434 ×0.29 | 1 005 ×0.46 | 560 ×0.70 | 247 ×0.83 |

Profil : 33 dispatchs d'attention à pp2048 pour 6 couches, 27 de la mesure au chargement ; 0.818 du profil de pp2048 retiré, noyau par noyau, de chaque longueur.

| profil (base) | pp2048 | pp8192 | pp24576 | pp61440 |
| --- | ---: | ---: | ---: | ---: |
| attention (sdpa_*) | 36.9 % | 74.9 % | 90.4 % | 96.3 % |
| GEMM (gemm_*) | 45.1 % | 17.5 % | 6.7 % | 2.6 % |
| DeltaNet (dn_*) | 14.4 % | 6.3 % | 2.4 % | 0.9 % |
| reste | 3.6 % | 1.2 % | 0.5 % | 0.2 % |
| GPU total (s) | 0.18 | 3.70 | 31.01 | 200.46 |
| attention (s) | 0.066 | 2.768 | 28.034 | 192.975 |
| attention, TFLOPS | 1.56 | 0.60 | 0.53 | 0.48 |

pp2048 : sdpa_flash_h8kv2d256_q16k16 36.9 % ; gemm_mm2d_1024x7168_bf16_t64_mf 14.5 % ; gemm_mm2d_1024x8224_bf16_t64_mf 12.4 % ; gemm_mm2d_3584x1024_res_t64_mf 10.2 %
pp8192 : sdpa_flash_h8kv2d256_q16k16 74.9 % ; gemm_mm2d_1024x7168_bf16_t64_mf 5.5 % ; gemm_mm2d_1024x8224_bf16_t64_mf 4.9 % ; gemm_mm2d_3584x1024_res_t64_mf 4.0 %
pp24576 : sdpa_flash_h8kv2d256_q16k16 90.4 % ; gemm_mm2d_1024x7168_bf16_t64_mf 2.2 % ; gemm_mm2d_1024x8224_bf16_t64_mf 1.9 % ; gemm_mm2d_3584x1024_res_t64_mf 1.5 %
pp61440 : sdpa_flash_h8kv2d256_q16k16 96.3 % ; gemm_mm2d_1024x7168_bf16_t64_mf 0.8 % ; gemm_mm2d_1024x8224_bf16_t64_mf 0.7 % ; gemm_mm2d_3584x1024_res_t64_mf 0.6 %

