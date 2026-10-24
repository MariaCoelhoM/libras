"""Junta todos os resultados/<experimento>/metricas.json numa tabela comparativa.

    python resumo_resultados.py

Grava resultados/resumo.md (para colar no texto) e resultados/resumo.csv.
O desvio mostrado e o ENTRE DOBRAS (quem esta no teste muda) quando o experimento tem
mais de uma dobra; senao, o ENTRE SEMENTES (so o sorteio do modelo muda).
A coluna 'desvio' diz qual dos dois e.
"""

import argparse
import csv
import glob
import json
import os


def pick_deviation(stats):
    if stats["desvio_entre_dobras"] is not None:
        return stats["desvio_entre_dobras"], "dobras"
    if stats["desvio_entre_sementes"] is not None:
        return stats["desvio_entre_sementes"], "sementes"
    return None, "-"


def fmt(stats):
    sd, _ = pick_deviation(stats)
    return f"{stats['media']:.3f}" + (f" ± {sd:.3f}" if sd is not None else "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="resultados")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.results_dir, "*", "metricas.json")))
    if not paths:
        raise SystemExit(f"Nenhum metricas.json em {args.results_dir}/*/ - rode os treinos primeiro.")

    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        r = m["resumo"]
        _, kind = pick_deviation(r["acuracia"])
        rows.append({
            "experimento": m["experimento"],
            "dobras": r["n_dobras"],
            "sementes": r["n_sementes"],
            "acuracia": fmt(r["acuracia"]),
            "f1_macro": fmt(r["f1_macro"]),
            "desvio": kind,
        })

    headers = ["experimento", "dobras", "sementes", "acuracia", "f1_macro", "desvio"]
    titles = ["Experimento", "Dobras", "Sementes", "Acurácia", "F1 macro", "Desvio"]

    lines = ["| " + " | ".join(titles) + " |", "|" + "|".join(["---"] * len(titles)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    table = "\n".join(lines)

    print(table)

    with open(os.path.join(args.results_dir, "resumo.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n\nMédia ± desvio. 'Desvio' indica de onde vem o ± (dobras: muda quem está no teste; "
                        "sementes: muda só o sorteio do modelo). Divisão sempre por sinalizador "
                        "(alfabeto: divisão original train/test do dataset).\n")
    with open(os.path.join(args.results_dir, "resumo.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSalvo em {args.results_dir}/resumo.md e resumo.csv")


if __name__ == "__main__":
    main()
