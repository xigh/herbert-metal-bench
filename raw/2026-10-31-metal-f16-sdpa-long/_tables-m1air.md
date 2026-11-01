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

