"""Funcoes compartilhadas entre os scripts de treino (train_lstm.py,
train_lstm_hands_face.py, train_cnn.py, matriz.py): filtro de classes,
identificacao do sinalizador e contagem de amostras por divisao.
"""

import csv
from collections import Counter

import numpy as np


def load_class_filter(classes_arg):
    """--classes pode ser uma lista separada por virgula, ou @arquivo.txt (uma classe por linha)."""
    if classes_arg is None:
        return None
    if classes_arg.startswith("@"):
        with open(classes_arg[1:], encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    return {c.strip() for c in classes_arg.split(",") if c.strip()}


def extract_signer_ids(paths, signer_regex):
    ids = []
    for p in paths:
        match = signer_regex.search(str(p))
        ids.append(match.group(1) if match else "desconhecido")
    ids_arr = np.array(ids)
    if "desconhecido" in ids_arr:
        n = int((ids_arr == "desconhecido").sum())
        print(f"Aviso: {n} arquivo(s) sem sinalizador identificavel pelo regex - "
              f"tratados como um grupo unico 'desconhecido'. Ajuste --signer_regex se isso nao fizer sentido.")
    return ids_arr


def log_counts(label, y_raw_subset, counts_by_split):
    counts = Counter(y_raw_subset.tolist())
    print(f"\nContagem de amostras - {label}:")
    for classe, n in sorted(counts.items()):
        print(f"  {classe}: {n}")
    counts_by_split[label] = counts


def save_counts_csv(path, counts_by_split):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["divisao", "classe", "quantidade"])
        for split_name, counts in counts_by_split.items():
            for classe, n in sorted(counts.items()):
                writer.writerow([split_name, classe, n])
    print(f"\nContagem de amostras salva em {path}")
