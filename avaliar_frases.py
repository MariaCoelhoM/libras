"""Camada linguistica: glosas (palavras reconhecidas) -> frase em portugues.

Gera a frase de ate tres jeitos e compara com a referencia por BLEU:
  (a) regras + spaCy: spaCy da a classe gramatical e o genero de cada palavra,
      as regras montam a frase (artigo, verbo, concordancia)
  (b) app: o MESMO tradutor que o app.py usa (frases.montar_frase: Groq com
      Gemini de reserva). So roda com --app e precisa das chaves de API.
  (c) LLM local via Ollama (opcional, so roda se voce passar --llm)

Para cada sistema o script mostra:
  - BLEU (sacreBLEU, corpus), com as precisoes de 1 a 4 gramas, a penalidade
    de brevidade (BP) e o intervalo de confianca de 95% por bootstrap
  - chrF e quantas frases sairam identicas a referencia
  - as frases com menor BLEU, para ver onde o sistema erra

Uso:
  python avaliar_frases.py
  python avaliar_frases.py --app
  python avaliar_frases.py --app --llm qwen2.5:3b
  python avaliar_frases.py --app --sem-caixa      # BLEU ignorando maiusculas

Instalacao:
  pip install spacy sacrebleu requests
  python -m spacy download pt_core_news_sm
"""

import argparse
import csv
import json
import os
import random
import time

from sacrebleu.metrics import BLEU, CHRF
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


# O app manda ao tradutor glosas em minusculas e COM acento (o rotulo do modelo e
# "America", a tela mostra "América" e o app envia "américa"). O CSV usa o rotulo
# do modelo em MAIUSCULAS e sem acento. Sem converter, montar_frase trataria
# a glosa em maiusculas como palavra soletrada, e o teste nao seria o do app.
ACENTOS_APP = {"america": "américa", "maca": "maçã", "nao": "não"}


def glosas_do_app(glosas):
    return [ACENTOS_APP.get(g.lower(), g.lower()) for g in glosas]


def gerar_app(glosas, pausa=0.0):
    """Roda o tradutor real do app. Devolve (frase, usou_llm, confere)."""
    from frases import montar_frase

    frase, confere, usou_llm = montar_frase(glosas_do_app(glosas))
    if pausa:
        time.sleep(pausa)
    return frase, usou_llm, confere


def bleu_por_frase(bleu, hipoteses, referencias):
    # sentence_score usa suavizacao: sem ela, frase curta sem 4-grama daria 0
    return [bleu.sentence_score(h, [r]).score for h, r in zip(hipoteses, referencias)]


def intervalo_bootstrap(bleu, hipoteses, referencias, n, semente=0):
    """IC de 95% do BLEU de corpus, sorteando as frases com reposicao.

    Com poucas frases (o CSV tem ~30) o BLEU balanca muito de um conjunto para
    outro, entao o numero sozinho engana. O intervalo mostra essa incerteza.
    """
    if n <= 0 or len(hipoteses) < 2:
        return None
    sorteio = random.Random(semente)
    indices = range(len(hipoteses))
    notas = []
    for _ in range(n):
        escolha = [sorteio.choice(indices) for _ in indices]
        notas.append(
            bleu.corpus_score([hipoteses[i] for i in escolha], [[referencias[i] for i in escolha]]).score
        )
    notas.sort()
    return notas[int(0.025 * (n - 1))], notas[int(0.975 * (n - 1))]


def avaliar(nome, hipoteses, referencias, glosas, args):
    """Imprime as metricas de um sistema e devolve um dicionario para o JSON."""
    bleu = BLEU(lowercase=args.sem_caixa)
    chrf = CHRF()
    resultado = bleu.corpus_score(hipoteses, [referencias])
    pontos = bleu_por_frase(bleu, hipoteses, referencias)
    exatas = sum(h.strip().lower() == r.strip().lower() for h, r in zip(hipoteses, referencias))
    ic = intervalo_bootstrap(bleu, hipoteses, referencias, args.bootstrap)

    prec = "/".join(f"{p:.1f}" for p in resultado.precisions)
    faixa = f" (IC95% {ic[0]:.1f}-{ic[1]:.1f})" if ic else ""
    print(f"{nome:8} BLEU {resultado.score:5.1f}{faixa} | chrF {chrf.corpus_score(hipoteses, [referencias]).score:5.1f}"
          f" | identicas {exatas}/{len(referencias)}")
    print(f"{'':8} precisoes 1-4 gramas {prec} | BP {resultado.bp:.3f} | tamanho hipotese/referencia {resultado.ratio:.3f}")

    if args.piores:
        ordem = sorted(range(len(pontos)), key=lambda i: pontos[i])[: args.piores]
        print(f"{'':8} menores BLEU por frase:")
        for i in ordem:
            if pontos[i] >= 100:
                break  # o resto e identico
            print(f"{'':10} {pontos[i]:5.1f}  {' '.join(glosas[i])}  ->  {hipoteses[i]}   [ref: {referencias[i]}]")

    return {
        "bleu": round(resultado.score, 2),
        "ic95": [round(ic[0], 2), round(ic[1], 2)] if ic else None,
        "precisoes": [round(p, 2) for p in resultado.precisions],
        "bp": round(resultado.bp, 4),
        "razao_tamanho": round(resultado.ratio, 4),
        "chrf": round(chrf.corpus_score(hipoteses, [referencias]).score, 2),
        "identicas": exatas,
        "frases": len(referencias),
        "assinatura": str(bleu.get_signature()),
        "bleu_por_frase": [round(p, 2) for p in pontos],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="glosas_referencia.csv", help="colunas: glosas (separadas por espaco), referencia")
    parser.add_argument("--llm", default=None, help="nome do modelo no Ollama (ex. qwen2.5:3b). Sem isso, so roda regras+spaCy.")
    parser.add_argument("--app", action="store_true", help="avalia o tradutor do app (frases.montar_frase: Groq/Gemini). Precisa das chaves de API.")
    parser.add_argument("--pausa", type=float, default=0.0, help="segundos de espera entre as chamadas do --app (evita limite de requisicoes)")
    parser.add_argument("--sem-caixa", action="store_true", help="BLEU sem diferenciar maiusculas de minusculas")
    parser.add_argument("--bootstrap", type=int, default=1000, help="reamostragens para o intervalo de confianca do BLEU (0 desliga)")
    parser.add_argument("--piores", type=int, default=5, help="quantas frases de menor BLEU listar por sistema (0 desliga)")
    parser.add_argument("--out", default="resultados_frases.csv")
    parser.add_argument("--json", default="resultados_bleu.json", help="onde salvar as metricas em JSON")
    args = parser.parse_args()

    if args.app:
        try:  # o app le GROQ_API_KEY / GEMINI_API_KEY do ambiente; aceita tambem um arquivo .env
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

    with open(args.csv, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    glosas = [l["glosas"].split() for l in linhas]
    referencias = [l["referencia"] for l in linhas]

    regras = [gerar_regras(g) for g in glosas]

    app, app_llm, app_confere = [], [], []
    if args.app:
        for g in glosas:
            frase, usou_llm, confere = gerar_app(g, args.pausa)
            app.append(frase)
            app_llm.append(usou_llm)
            app_confere.append(confere)

    llm, tempos = [], []
    if args.llm:
        for g in glosas:
            inicio = time.perf_counter()
            llm.append(gerar_llm(g, args.llm))
            tempos.append((time.perf_counter() - inicio) * 1000)

    print(f"\n{len(linhas)} sequencias avaliadas | BLEU {'sem' if args.sem_caixa else 'com'} diferenciar maiusculas\n")
    sistemas = {"regras": avaliar("regras", regras, referencias, glosas, args)}

    if app:
        sistemas["app"] = avaliar("app", app, referencias, glosas, args)
        reserva = app_llm.count(False)
        if reserva:
            # Sem LLM, montar_frase devolve so as palavras coladas. Isso nao e traducao e
            # derruba o BLEU por motivo de rede/chave, nao de qualidade: mostra separado.
            print(f"\n*** ATENCAO: {reserva}/{len(app)} frases sairam SEM LLM (Groq e Gemini falharam: chave, cota ou rede).")
            print("    O BLEU do app acima inclui essas frases. Abaixo, so as que o LLM respondeu:\n")
            ok = [i for i, usou in enumerate(app_llm) if usou]
            if len(ok) >= 2:
                sistemas["app_so_llm"] = avaliar(
                    "app(LLM)", [app[i] for i in ok], [referencias[i] for i in ok], [glosas[i] for i in ok], args
                )
        sem_confere = app_confere.count(False) - reserva
        if sem_confere > 0:
            print(f"\n{sem_confere} frase(s) do app foram marcadas pelo proprio app como suspeitas (falta palavra ou sujeito invalido).")

    if llm:
        sistemas["llm"] = avaliar("llm", llm, referencias, glosas, args)
        print(f"\nLatencia do LLM por frase: media {sum(tempos) / len(tempos):.0f} ms, maior {max(tempos):.0f} ms")

    # Planilha para a avaliacao humana (nota de 1 a 5 para cada frase), ja com o BLEU de cada frase
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow([
            "glosas", "referencia", "regras", "app", "llm",
            "bleu_regras", "bleu_app", "bleu_llm", "app_usou_llm", "app_confere",
            "nota_humana_regras", "nota_humana_app", "nota_humana_llm",
        ])
        for i, l in enumerate(linhas):
            escritor.writerow([
                l["glosas"], referencias[i], regras[i],
                app[i] if app else "", llm[i] if llm else "",
                sistemas["regras"]["bleu_por_frase"][i],
                sistemas["app"]["bleu_por_frase"][i] if app else "",
                sistemas["llm"]["bleu_por_frase"][i] if llm else "",
                app_llm[i] if app else "", app_confere[i] if app else "",
                "", "", "",
            ])
    print(f"\nPlanilha salva em {args.out}")

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump({
            "csv": os.path.basename(args.csv),
            "frases": len(linhas),
            "sem_caixa": args.sem_caixa,
            "sistemas": sistemas,
        }, f, ensure_ascii=False, indent=2)
    print(f"Metricas salvas em {args.json}")


if __name__ == "__main__":
    main()