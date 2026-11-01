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

