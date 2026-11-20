"""Tabelas dos resultados de palavras: por modelo e por origem (MINDS x Nucleo Comum).

    python tabela_resultados.py

Le resultados/<experimento>/metricas.json e relatorio.txt (LSTM e CNN) e
resultados/lstm_treino_val/baseline_*.json (Random Forest e k-NN).
Grava resultados/tabela_modelos.md (para colar no texto) e resultados/tabela_modelos.csv.

Todos os modelos de palavras sao avaliados no MESMO teste (235 amostras, divisao por
sinalizador). Atencao a coluna "n treino": a etapa 1 da LSTM treina so com o treino
(sem a validacao), entao ela nao e comparavel em tamanho de treino com a LSTM final e
com os baselines.
"""

import argparse
import csv
import json
import os
import re

import numpy as np

ORIGENS = ("minds", "nucleo")


def carregar(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def por_origem_do_relatorio(path):
    """Acuracia por origem lida do relatorio.txt do treino (linhas 'Acuracia no teste (minds, 200 amostras): 0.7450')."""
    resultado = {}
    if not os.path.exists(path):
        return resultado
    with open(path, encoding="utf-8") as f:
        for linha in f:
            m = re.match(r"Acuracia no teste \((\w+), (\d+) amostras\): ([0-9.]+)", linha)
            if m:
                resultado[m.group(1)] = (float(m.group(3)), int(m.group(2)))
    return resultado


def linha_lstm(results_dir, experimento, rotulo):
    pasta = os.path.join(results_dir, experimento)
    m = carregar(os.path.join(pasta, "metricas.json"))
    r = m["resumo"]
    rodada = m["por_rodada"][0]
    origem = por_origem_do_relatorio(os.path.join(pasta, "relatorio.txt"))
    return {
        "modelo": rotulo,
        "n_treino": rodada.get("n_treino"),
        "n_teste": rodada["n_teste"],
        "acuracia": (r["acuracia"]["media"], None),
        "f1_macro": (r["f1_macro"]["media"], None),
        "origem": {k: v[0] for k, v in origem.items()},
        "n_origem": {k: v[1] for k, v in origem.items()},
    }


def linha_baseline(results_dir, arquivo, rotulo, split_path):
    j = carregar(os.path.join(results_dir, arquivo))
    runs = j["execucoes"]
    origem = {}
    for o in ORIGENS:
        vals = [r["acuracia_por_origem"][o] for r in runs if o in r.get("acuracia_por_origem", {})]
        if vals:
            origem[o] = float(np.mean(vals))
    s = np.load(split_path, allow_pickle=True)
    return {
        "modelo": rotulo,
        "n_treino": int(len(s["idx_treino"]) + len(s["idx_val"])),
        "n_teste": int(len(s["idx_teste"])),
        "acuracia": (j["acuracia_media"], j["acuracia_desvio"]),
        "f1_macro": (j["f1_macro_medio"], j["f1_macro_desvio"]),
        "origem": origem,
        "n_origem": {},
    }


def fmt(par, virgula=True):
    media, desvio = par
    texto = f"{media:.3f}" + (f" ± {desvio:.3f}" if desvio else "")
    return texto.replace(".", ",") if virgula else texto


def fmt_origem(linha, o, virgula=True):
    if o not in linha["origem"]:
        return "-"
    texto = f"{linha['origem'][o]:.3f}"
    return texto.replace(".", ",") if virgula else texto


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="resultados")
    parser.add_argument("--final", default="lstm_treino_val",
                        help="experimento do modelo final (dele vem o teste dos baselines)")
    args = parser.parse_args()
    rd = args.results_dir

    split_final = os.path.join(rd, args.final, "split.npz")
    linhas = [
        linha_lstm(rd, args.final, "LSTM final (treino+validacao, 19 epocas)"),
        linha_baseline(rd, os.path.join(args.final, "baseline_rf.json"), "Random Forest (baseline)", split_final),
        linha_baseline(rd, os.path.join(args.final, "baseline_knn.json"), "k-NN (baseline)", split_final),
        linha_lstm(rd, "lstm_unificado", "LSTM etapa 1 (so treino, EarlyStopping)"),
        linha_lstm(rd, "lstm_unificado_semPesos", "LSTM etapa 1 sem pesos de classe"),
    ]

    n_minds = next((l["n_origem"]["minds"] for l in linhas if "minds" in l["n_origem"]), "?")
    n_nucleo = next((l["n_origem"]["nucleo"] for l in linhas if "nucleo" in l["n_origem"]), "?")
    titulos = ["Modelo", "n treino", "n teste", "Acurácia", "F1 macro",
               f"MINDS-Libras (n={n_minds})", f"V-Librasil Núcleo Comum (n={n_nucleo})"]

    md = ["| " + " | ".join(titulos) + " |", "|" + "|".join(["---"] * len(titulos)) + "|"]
    for l in linhas:
        md.append("| " + " | ".join([
            l["modelo"], str(l["n_treino"]), str(l["n_teste"]), fmt(l["acuracia"]), fmt(l["f1_macro"]),
            fmt_origem(l, "minds"), fmt_origem(l, "nucleo"),
        ]) + " |")

    alfabeto = os.path.join(rd, "modelo_alfabeto", "metricas.json")
    nota_alfabeto = ""
    if os.path.exists(alfabeto):
        a = carregar(alfabeto)
        r = a["resumo"]
        rod = a["por_rodada"][0]
        nota_alfabeto = (
            f"\nAlfabeto (CNN, imagens, divisão original train/test do dataset, sem sinalizador): "
            f"acurácia {fmt((r['acuracia']['media'], r['acuracia']['desvio_entre_sementes']))} "
            f"em {rod['n_teste']} imagens de teste (média de {r['n_sementes']} sementes).\n"
        )

    texto = "\n".join(md)
    print(texto)
    print(nota_alfabeto)

    with open(os.path.join(rd, "tabela_modelos.md"), "w", encoding="utf-8") as f:
        f.write(texto + "\n\n"
                "Palavras: mesmo teste para todos os modelos (divisão por sinalizador, por base). "
                "Média ± desvio entre sementes quando há mais de uma (Random Forest); a LSTM tem uma semente. "
                "Com 35 amostras no Núcleo Comum, as diferenças nessa coluna são frágeis.\n"
                + nota_alfabeto)

    campos = ["modelo", "n_treino", "n_teste", "acuracia", "f1_macro", "minds", "nucleo"]
    with open(os.path.join(rd, "tabela_modelos.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for l in linhas:
            w.writerow({
                "modelo": l["modelo"], "n_treino": l["n_treino"], "n_teste": l["n_teste"],
                "acuracia": fmt(l["acuracia"], False), "f1_macro": fmt(l["f1_macro"], False),
                "minds": fmt_origem(l, "minds", False), "nucleo": fmt_origem(l, "nucleo", False),
            })
    print(f"Salvo em {rd}/tabela_modelos.md e tabela_modelos.csv")


if __name__ == "__main__":
    main()
