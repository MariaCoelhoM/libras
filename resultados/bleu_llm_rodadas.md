# BLEU e chrF do sistema com LLM (glosas_teste.csv, 26 frases)

Sistema `app`: spaCy + Groq (`openai/gpt-oss-20b`), com Gemini como reserva.
Comando: `python avaliar_frases.py --csv glosas_teste.csv --app --pausa 3`, com a
versão final do `frases.py` (repetição no Groq quando a resposta vem vazia e descarte
de respostas que não são tradução). Três execuções independentes, em processos
separados. O LLM não é determinístico: o resultado varia entre execuções.

Referência determinística (regras + spaCy, sem LLM): BLEU 61,9 (IC95% 43,3–79,7),
chrF 79,9 (IC95% 70,2–89,2), 13/26 frases idênticas.

| Rodada | BLEU (IC95%) | chrF (IC95%) | Idênticas | Frases sem LLM (fallback) | Tentativas do Groq com resposta vazia | Respostas inválidas descartadas |
|---|---|---|---|---|---|---|
| 1 | 80,3 (64,3–94,6) | 87,7 (77,9–96,4) | 20/26 | 0 | 3 | 0 |
| 2 | 84,2 (68,7–95,6) | 89,5 (80,1–96,8) | 21/26 | 1 (`não`) | 4 | 3 |
| 3 | 78,1 (61,5–92,2) | 85,9 (75,5–94,6) | 19/26 | 0 | 6 | 3 |
| Média ± desvio entre rodadas | 80,9 ± 3,1 | 87,7 ± 1,8 | 20,0 | | | |

Notas:

- O IC95% de cada linha é o calculado pelo `avaliar_frases.py` dentro da rodada. O
  desvio da última linha é a variação entre rodadas (efeito do LLM) e é outra coisa.
- Rodada 2, só com as 25 frases que o LLM respondeu: BLEU 84,1, chrF 89,4.
- "Respostas inválidas" são respostas que não traduzem (por exemplo, "não há sinais
  para traduzir"). O `frases.py` as descarta e pede outra; se nenhuma servir, a frase
  sai pelo fallback local e é contada como "sem LLM".
- Frases de menor BLEU nas três rodadas: `cinco filho medo` (8,6: "O filho tem medo
  de cinco" contra "Os cinco filhos têm medo"); `cinco banco` (10,7 nas rodadas 1 e
  3); `banco esquina` (38,0 nas rodadas 1 e 3: "fica" em vez de "está"); e as
  perguntas com `onde` (42,7: "Onde fica o banheiro?" contra "Onde está o
  banheiro?"). Três grupos de erro: convenção de redação ("fica"/"está"), numeral
  com plural, e falhas de infraestrutura (Groq vazio, Gemini sem cota).
- `resultados/bleu_glosas_teste.json` e `resultados/frases_glosas_teste.csv`
  correspondem à última execução (rodada 3).
