"""Camada linguistica: glosas (palavras reconhecidas) -> frase em portugues.

Gera a frase de dois jeitos e compara com a referencia por BLEU:
  (a) regras + spaCy: spaCy da a classe gramatical e o genero de cada palavra,
      as regras montam a frase (artigo, verbo, concordancia)
  (b) LLM local via Ollama (opcional, so roda se voce passar --llm)

Uso:
  python avaliar_frases.py
  python avaliar_frases.py --llm qwen2.5:3b

Instalacao:
  pip install spacy sacrebleu requests
  python -m spacy download pt_core_news_sm
"""

import argparse
import csv
import time

import sacrebleu
import spacy

nlp = spacy.load("pt_core_news_sm")

# O spaCy erra estas duas palavras do vocabulario quando analisadas sozinhas
# (America vira adjetivo, Vontade vira adjetivo masculino).
CORRECOES = {"america": ("PROPN", "Fem"), "vontade": ("NOUN", "Fem")}
# Os rotulos do modelo nao tem acento
EXIBIR = {"america": "América", "maca": "maçã"}

ARTIGO = {"Masc": "o", "Fem": "a"}
ARTIGO_PL = {"Masc": "os", "Fem": "as"}
DE = {"Masc": "do", "Fem": "da"}


def analisar(glosa):
    palavra = glosa.lower()
    token = nlp(palavra)[0]
    if palavra in CORRECOES:
        pos, genero = CORRECOES[palavra]
    else:
        generos = token.morph.get("Gender")
        pos, genero = token.pos_, (generos[0] if generos else "Masc")
    return {"palavra": EXIBIR.get(palavra, palavra), "base": palavra, "pos": pos, "genero": genero}


def pluralizar(palavra):
    return palavra[:-1] + "ns" if palavra.endswith("m") else palavra + "s"


def conjugar(infinitivo, plural):
    # presente, 3a pessoa (verbos regulares em -ar e -er)
    radical = infinitivo[:-2]
    if infinitivo.endswith("ar"):
        return radical + ("am" if plural else "a")
    return radical + ("em" if plural else "e")


def concordar(adjetivo, genero, plural):
    if adjetivo.endswith("o"):
        adjetivo = adjetivo[:-1] + ("a" if genero == "Fem" else "o")
    return pluralizar(adjetivo) if plural else adjetivo


def gerar_regras(glosas):
    toks = [analisar(g) for g in glosas]

    numeral = None
    if toks[0]["pos"] == "NUM":
        numeral = toks[0]["palavra"]
        toks = toks[1:]

    if not toks:
        return numeral.capitalize() + "."

    sujeito = toks[0]
    plural = numeral is not None
    nome = pluralizar(sujeito["palavra"]) if plural else sujeito["palavra"]
    artigo = (ARTIGO_PL if plural else ARTIGO)[sujeito["genero"]]
    sn = artigo + " " + (numeral + " " if plural else "") + nome  # "as cinco balas"
    ser = "são" if plural else "é"
    resto = toks[1:]
    pos = [t["pos"] for t in resto]

    if not resto:
        frase = (numeral + " " + nome) if plural else sn
    elif pos == ["ADJ"]:
        frase = f"{sn} {ser} {concordar(resto[0]['palavra'], sujeito['genero'], plural)}"
    elif pos == ["VERB"]:
        frase = f"{sn} {conjugar(resto[0]['base'], plural)}"
    elif resto[0]["base"] in ("medo", "vontade") and len(resto) == 1:
        frase = f"{sn} {'têm' if plural else 'tem'} {resto[0]['palavra']}"
    elif resto[0]["base"] == "vontade" and pos[1:] == ["VERB"]:
        frase = f"{sn} {'têm' if plural else 'tem'} vontade de {resto[1]['base']}"
    elif len(resto) == 2 and resto[1]["base"] == "medo":
        objeto = resto[0]
        frase = f"{sn} {'têm' if plural else 'tem'} medo {DE[objeto['genero']]} {objeto['palavra']}"
    elif len(resto) == 2 and pos[1] == "VERB":
        # Libras costuma ser sujeito-objeto-verbo; o portugues, sujeito-verbo-objeto
        objeto = resto[0]
        frase = f"{sn} {conjugar(resto[1]['base'], plural)} {ARTIGO[objeto['genero']]} {objeto['palavra']}"
    else:
        frase = " ".join([sn] + [t["palavra"] for t in resto])  # sem regra: so junta as palavras

    return frase[0].upper() + frase[1:] + "."


def gerar_llm(glosas, modelo):
    import requests

    prompt = (
        "Voce transforma glosas de Libras em uma frase curta em portugues. "
        "Use as palavras dadas, adicionando so artigos, preposicoes e a conjugacao do verbo. "
        "Responda apenas com a frase.\n\n"
        "Glosas: MENINO BOLA CHUTAR\nFrase: O menino chuta a bola.\n\n"
        "Glosas: CASA GRANDE\nFrase: A casa é grande.\n\n"
        f"Glosas: {' '.join(glosas)}\nFrase:"
    )
    resposta = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": modelo, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
        timeout=120,
    )
    resposta.raise_for_status()
    return resposta.json()["response"].strip().split("\n")[0]


def avaliar(nome, hipoteses, referencias):
    bleu = sacrebleu.corpus_bleu(hipoteses, [referencias])
    chrf = sacrebleu.corpus_chrf(hipoteses, [referencias])
    exatas = sum(h.strip().lower() == r.strip().lower() for h, r in zip(hipoteses, referencias))
    print(f"{nome:10} BLEU {bleu.score:5.1f} | chrF {chrf.score:5.1f} | frases identicas {exatas}/{len(referencias)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="glosas_referencia.csv", help="colunas: glosas (separadas por espaco), referencia")
    parser.add_argument("--llm", default=None, help="nome do modelo no Ollama (ex. qwen2.5:3b). Sem isso, so roda regras+spaCy.")
    parser.add_argument("--out", default="resultados_frases.csv")
    args = parser.parse_args()

    with open(args.csv, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    glosas = [l["glosas"].split() for l in linhas]
    referencias = [l["referencia"] for l in linhas]

    regras = [gerar_regras(g) for g in glosas]

    llm, tempos = [], []
    if args.llm:
        for g in glosas:
            inicio = time.perf_counter()
            llm.append(gerar_llm(g, args.llm))
            tempos.append((time.perf_counter() - inicio) * 1000)

    print(f"\n{len(linhas)} sequencias avaliadas\n")
    avaliar("regras", regras, referencias)
    if llm:
        avaliar("llm", llm, referencias)
        print(f"\nLatencia do LLM por frase: media {sum(tempos) / len(tempos):.0f} ms, maior {max(tempos):.0f} ms")

    # Planilha para a avaliacao humana (nota de 1 a 5 para cada frase)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["glosas", "referencia", "regras", "llm", "nota_humana_regras", "nota_humana_llm"])
        for i, l in enumerate(linhas):
            escritor.writerow([l["glosas"], referencias[i], regras[i], llm[i] if llm else "", "", ""])
    print(f"\nPlanilha salva em {args.out}")


if __name__ == "__main__":
    main()
