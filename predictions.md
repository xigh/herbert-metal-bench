# Prédiction

Une seule prédiction a été écrite avant la mesure des prompts longs, en lisant les chiffres de la passe longue de `raw/2026-10-31-metal-f16-vs-engines`, avant les mesures de `raw/2026-10-31-metal-f16-sdpa-long` :

> « on se fait déboiter en prefill… mon intuition, c'est qu'au-delà d'un certain pp il faut utiliser les simdgroups et autres accélérations matérielles, sur le M1, M3 Ultra et M5 »

Philippe, 2026-10-31.

Ce que la mesure a rendu est dans `raw/2026-10-31-metal-f16-sdpa-long/README.md` (la part de l'attention scalaire dans le prefill, puis l'attention réécrite sur les unités matricielles, sur les trois machines) et dans `raw/2026-11-01-metal-f16-art004/README.md` (le prefill bf16 jusqu'à 81 920 tokens sur le M3 Ultra, au commit publié).
