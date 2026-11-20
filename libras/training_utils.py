"""Funcoes compartilhadas entre os scripts de treino (train_lstm.py,
train_cnn.py, train_baseline.py, matriz.py): filtro de classes,
identificacao do sinalizador, divisao por sinalizador e contagem de
amostras por divisao.
"""

import csv
import re
from collections import Counter

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# MINDS-Libras: 01AcontecerSinalizador01-5.mp4  -> sinalizador 01
# V-Librasil:   Não_Articulador20.mp4           -> articulador (sinalizador) 20
REGEX_SINALIZADOR_MINDS = r"Sinalizador(\d+)"
REGEX_SINALIZADOR_NUCLEO = r"Articulador(\d+)"


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


# ---------------- Divisao por sinalizador (base unificada) ----------------

def signer_ids_unificado(paths, origem,
                         regex_minds=REGEX_SINALIZADOR_MINDS,
                         regex_nucleo=REGEX_SINALIZADOR_NUCLEO):
    """Identificador de grupo de cada amostra: '<origem>:<sinalizador>'.

    O prefixo da origem evita que o sinalizador 01 do MINDS-Libras seja confundido
    com o articulador 01 do V-Librasil. Se o nome do arquivo nao casar com o regex,
    a amostra vira um grupo proprio (e o numero de casos e avisado).
    """
    regex = {"minds": re.compile(regex_minds), "nucleo": re.compile(regex_nucleo)}
    ids, sem_id = [], 0
    for p, o in zip(paths, origem):
        o = str(o)
        m = regex[o].search(str(p)) if o in regex else None
        if m:
            ids.append(f"{o}:{m.group(1)}")
        else:
            ids.append(f"{o}:{p}")
            sem_id += 1
    if sem_id:
        print(f"Aviso: {sem_id} arquivo(s) sem sinalizador identificavel - cada um virou um grupo proprio.")
    return np.array(ids)


def split_por_sinalizador(signer_ids, origem, indices, fracao, seed, min_grupos=3):
    """Separa `indices` em (resto, separado) sem repartir nenhum sinalizador.

    A divisao e feita DENTRO de cada origem, para que as duas bases apareçam em
    cada lado. Uma origem com menos de `min_grupos` sinalizadores nao e dividida:
    fica toda em `resto` (nao da para tirar um sinalizador sem esvaziar o treino).
    `fracao` e a fracao de SINALIZADORES (nao de amostras) que vai para `separado`.
    Devolve dois arrays de indices (globais), ordenados.
    """
    indices = np.asarray(indices)
    origem = np.asarray(origem)
    if fracao <= 0:
        # Sem separacao (ex.: --val_size 0 para treinar com treino+validacao).
        return np.sort(indices), np.array([], dtype=int)
    resto, separado = [], []
    for o in np.unique(origem[indices]):
        sub = indices[origem[indices] == o]
        grupos = signer_ids[sub]
        if len(set(grupos.tolist())) < min_grupos:
            resto.append(sub)
            continue
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=fracao, random_state=seed)
                    .split(sub, groups=grupos))
        resto.append(sub[a])
        separado.append(sub[b])
    resto = np.sort(np.concatenate(resto)) if resto else np.array([], dtype=int)
    separado = np.sort(np.concatenate(separado)) if separado else np.array([], dtype=int)
    return resto, separado