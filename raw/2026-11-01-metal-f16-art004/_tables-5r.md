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
