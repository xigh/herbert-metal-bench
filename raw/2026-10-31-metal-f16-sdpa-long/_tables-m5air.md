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

