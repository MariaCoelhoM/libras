"""Infraestrutura de experimentos, compartilhada por train_cnn.py,
train_lstm_hands_face.py e train_baseline.py.

Cada treino repete (dobras x sementes) e grava tudo em
resultados/<experimento>/ :

    metricas.json            resumo (media +- desvio), cada rodada, metricas por classe
    matriz_confusao.csv      contagens somadas sobre todas as rodadas
    matriz_confusao.png      o mesmo, normalizado por classe real (so com poucas classes)
    confusoes_principais.csv os pares de sinais mais confundidos (base da analise de erros)
    argumentos.json          argumentos usados, versoes das bibliotecas
    contagem_amostras.csv    amostras por classe e por divisao

Duas fontes de variacao sao separadas de proposito:
  - entre dobras: muda QUEM esta no teste (sinalizadores diferentes);
  - entre sementes: muda so o sorteio do modelo (pesos iniciais, ordem dos lotes),
    com a divisao dos dados fixa.
"""

import csv
import json
import os
import platform
import random
from datetime import datetime
from importlib import metadata

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split

from libras.training_utils import save_counts_csv

# A divisao dos dados e SEMPRE a mesma (nao entra nas sementes), para que os
# experimentos sejam comparaveis entre si: so o modelo varia entre repeticoes.
SPLIT_SEED = 42

MAX_CLASSES_FOR_PNG = 60


def _version(*packages):
    """Versao instalada do primeiro nome que existir (o TensorFlow aparece como
    tensorflow, tensorflow-cpu ou tensorflow-intel conforme a instalacao)."""
    for package in packages:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return None


# ---------------- Sementes ----------------

def set_seed(seed, deterministic=False, use_tensorflow=True):
    """Fixa as sementes de random, NumPy e TensorFlow.

    deterministic=True tambem pede operacoes deterministicas ao TensorFlow
    (mesma semente -> mesmo resultado exato, mais lento). Sem isso, a mesma
    semente ainda reduz muito a variacao, mas contas em varias threads da CPU
    podem diferir nas ultimas casas decimais.
    """
    random.seed(seed)
    np.random.seed(seed)
    if not use_tensorflow:
        return
    try:
        import tensorflow as tf
    except ImportError:
        return
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()


# ---------------- Divisoes ----------------

def signer_splits(signer_ids, folds, test_size):
    """Divisoes por sinalizador (a mesma pessoa nunca esta em treino e teste).

    folds == 0 : uma divisao unica, test_size = fracao de sinalizadores no teste
                 (a mesma de antes: GroupShuffleSplit, random_state=42).
    folds >= 2 : validacao cruzada por sinalizador (GroupKFold). folds igual ao
                 numero de sinalizadores = deixar um de fora por rodada.
    Devolve uma lista de (train_idx, test_idx).
    """
    indices = np.arange(len(signer_ids))
    n_signers = len(set(signer_ids.tolist()))

    if folds == 0:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SPLIT_SEED)
        return list(splitter.split(indices, groups=signer_ids))

    if folds < 2:
        raise SystemExit("--folds deve ser 0 (divisao unica) ou >= 2.")
    if folds > n_signers:
        raise SystemExit(f"--folds {folds} e maior que o numero de sinalizadores ({n_signers}).")
    return list(GroupKFold(n_splits=folds).split(indices, groups=signer_ids))


def train_val_split(train_idx, signer_ids, val_split):
    """Separa uma validacao (por sinalizador) de dentro do treino, para early stopping.

    Devolve (train_idx, val_idx). val_idx e None quando nao ha como validar:
    val_split == 0, ou menos de 3 sinalizadores no treino (tirar um deixaria
    quase nada para treinar; ex.: V-Librasil com 3 sinalizadores).
    """
    groups = signer_ids[train_idx]
    if val_split <= 0 or len(set(groups.tolist())) < 3:
        if val_split > 0:
            print("Aviso: poucos sinalizadores no treino - sem validacao separada; "
                  "treinando o numero fixo de epocas (sem early stopping).")
        return train_idx, None

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=SPLIT_SEED)
    sub_train, sub_val = next(splitter.split(train_idx, groups=groups))
    return train_idx[sub_train], train_idx[sub_val]


def static_splits(y, split_names, fallback_test_fraction=0.15):
    """Alfabeto: usa a divisao original do dataset (train/test) quando o .npz a tem;
    senao, cai num split estratificado. Devolve [(train_idx, test_idx)]."""
    if split_names is not None and {"train", "test"}.issubset(set(split_names.tolist())):
        print("Usando a divisao original train/test do dataset (nao re-sorteando amostras).")
        return [(np.where(split_names == "train")[0], np.where(split_names == "test")[0])]

    print("Aviso: o .npz nao tem a divisao original train/test - usando split aleatorio "
          "estratificado. Para usar a original, re-extraia com extract_landmarks_imagem.py atualizado.")
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=fallback_test_fraction, stratify=y, random_state=SPLIT_SEED)
    return [(train_idx, test_idx)]


def static_train_val(train_idx, y, val_split):
    """Validacao estratificada de dentro do treino (alfabeto, que nao tem sinalizador)."""
    if val_split <= 0:
        return train_idx, None
    sub_train, sub_val = train_test_split(
        train_idx, test_size=val_split, stratify=y[train_idx], random_state=SPLIT_SEED)
    return sub_train, sub_val


# ---------------- Metricas ----------------

def compute_metrics(y_true, y_pred):
    """Acuracia e F1 macro.

    O F1 macro e a media do F1 das classes que aparecem no teste daquela rodada.
    Classes sem nenhuma amostra no teste nao entram (senao contariam como F1 = 0
    sem o modelo ter errado nada).
    """
    present = np.unique(y_true)
    return {
        "acuracia": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
    }


def _mean_and_stds(runs, key):
    by_fold = {}
    for r in runs:
        by_fold.setdefault(r["dobra"], []).append(r[key])

    fold_means = np.array([np.mean(v) for v in by_fold.values()])
    seed_stds = [np.std(v, ddof=1) for v in by_fold.values() if len(v) > 1]
    return {
        "media": float(fold_means.mean()),
        "desvio_entre_dobras": float(fold_means.std(ddof=1)) if len(fold_means) > 1 else None,
        "desvio_entre_sementes": float(np.mean(seed_stds)) if seed_stds else None,
    }


def summarize_runs(runs):
    """Media e desvios sobre as rodadas. 'media' e a media das dobras (cada dobra ja
    promediada sobre as sementes), entao dobras com mais sementes nao pesam mais."""
    return {
        "n_dobras": len({r["dobra"] for r in runs}),
        "n_sementes": len({r["semente"] for r in runs}),
        "acuracia": _mean_and_stds(runs, "acuracia"),
        "f1_macro": _mean_and_stds(runs, "f1_macro"),
    }


# ---------------- Figuras ----------------

def plot_confusion(cm, class_names, path, title="Matriz de confusao"):
    """PNG da matriz normalizada por classe real (cada linha soma 1)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros(cm.shape, dtype=float), where=row_sums > 0)

    n = len(class_names)
    size = max(8, 0.5 * n)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, label="fracao da classe real")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    for i in range(n):
        for j in range(n):
            if norm[i, j] >= 0.01:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------- Registro de um experimento ----------------

class ExperimentLog:
    """Acumula as rodadas de um experimento e grava tudo no final."""

    def __init__(self, name, class_names):
        self.name = name
        self.class_names = [str(c) for c in class_names]
        self.n_classes = len(self.class_names)
        self.runs = []
        self.y_true_all, self.y_pred_all = [], []

    def add_run(self, fold, seed, y_true, y_pred, **info):
        """Registra uma rodada (uma dobra com uma semente). fold e 0-based."""
        run = {"dobra": int(fold), "semente": int(seed),
               "n_teste": int(len(y_true)), **compute_metrics(y_true, y_pred), **info}
        self.runs.append(run)
        self.y_true_all.append(np.asarray(y_true))
        self.y_pred_all.append(np.asarray(y_pred))
        print(f"[{self.name}] dobra {fold + 1}, semente {seed}: "
              f"acuracia={run['acuracia']:.4f}  f1_macro={run['f1_macro']:.4f}", flush=True)
        return run

    def _pooled(self):
        return np.concatenate(self.y_true_all), np.concatenate(self.y_pred_all)

    def print_report(self):
        y_true, y_pred = self._pooled()
        labels = np.arange(self.n_classes)
        print("\nRelatorio de classificacao (todas as rodadas somadas):")
        print(classification_report(y_true, y_pred, labels=labels,
                                   target_names=self.class_names, zero_division=0))

    def save(self, results_dir, args_dict, counts_by_split):
        out_dir = os.path.join(results_dir, self.name)
        if os.path.exists(os.path.join(out_dir, "metricas.json")):
            print(f"Aviso: {out_dir} ja tinha resultados - sobrescrevendo.")
        os.makedirs(out_dir, exist_ok=True)

        y_true, y_pred = self._pooled()
        labels = np.arange(self.n_classes)
        summary = summarize_runs(self.runs)

        prec, rec, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0)
        per_class = {
            name: {"precisao": float(prec[i]), "revocacao": float(rec[i]),
                   "f1": float(f1[i]), "suporte": int(support[i])}
            for i, name in enumerate(self.class_names)
        }

        with open(os.path.join(out_dir, "metricas.json"), "w", encoding="utf-8") as f:
            json.dump({"experimento": self.name, "resumo": summary,
                       "por_rodada": self.runs, "por_classe": per_class,
                       "observacao": "por_classe e as matrizes somam todas as rodadas (dobras x sementes)."},
                      f, indent=2, ensure_ascii=False)

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        with open(os.path.join(out_dir, "matriz_confusao.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["real\\previsto"] + self.class_names)
            for i, name in enumerate(self.class_names):
                writer.writerow([name] + cm[i].tolist())

        if self.n_classes <= MAX_CLASSES_FOR_PNG:
            plot_confusion(cm, self.class_names, os.path.join(out_dir, "matriz_confusao.png"),
                           title=f"Matriz de confusao - {self.name}")
        else:
            print(f"({self.n_classes} classes: PNG da matriz omitido por ilegivel; veja o CSV.)")

        pairs = [(self.class_names[i], self.class_names[j], int(cm[i, j]),
                  float(cm[i, j] / support[i]) if support[i] else 0.0)
                 for i in range(self.n_classes) for j in range(self.n_classes)
                 if i != j and cm[i, j] > 0]
        pairs.sort(key=lambda p: p[2], reverse=True)
        with open(os.path.join(out_dir, "confusoes_principais.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["real", "previsto", "contagem", "fracao_da_classe_real"])
            for row in pairs[:50]:
                writer.writerow([row[0], row[1], row[2], f"{row[3]:.4f}"])

        save_counts_csv(os.path.join(out_dir, "contagem_amostras.csv"), counts_by_split)

        with open(os.path.join(out_dir, "argumentos.json"), "w", encoding="utf-8") as f:
            json.dump({"argumentos": args_dict,
                       "versoes": {"python": platform.python_version(), "numpy": _version("numpy"),
                                   "scikit-learn": _version("scikit-learn"), "tensorflow": _version("tensorflow", "tensorflow-cpu", "tensorflow-intel")},
                       "gerado_em": datetime.now().isoformat(timespec="seconds")},
                      f, indent=2, ensure_ascii=False, default=str)

        def fmt(stats):
            sd = stats["desvio_entre_dobras"]
            kind = "entre dobras"
            if sd is None:
                sd, kind = stats["desvio_entre_sementes"], "entre sementes"
            return f"{stats['media']:.4f}" + (f" +- {sd:.4f} ({kind})" if sd is not None else "")

        print(f"\n=== {self.name}: {summary['n_dobras']} dobra(s) x {summary['n_sementes']} semente(s) ===")
        print(f"acuracia: {fmt(summary['acuracia'])}")
        print(f"f1 macro: {fmt(summary['f1_macro'])}")
        print(f"Resultados salvos em {out_dir}")
        return out_dir
