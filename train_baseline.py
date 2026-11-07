"""Baseline simples (Random Forest ou k-NN) para medir o ganho da CNN e da LSTM.

Usa os MESMOS vetores dos modelos neurais, so que achatados, e a mesma divisao:
  - .npz de video (X_hands, X_face_norm, X_traj, ...): sequencia (T x features)
    achatada em um vetor so. Divisao estratificada (test_size/seed iguais aos do
    treino da LSTM: 0.375 e 42) ou, se pedir --by_signer e o .npz tiver 'paths',
    por sinalizador.
  - .npz de alfabeto (chaves X e y): vetor de 21 x 3 = 63 valores; usa a divisao
    train/test do .npz se ela existir, senao uma divisao estratificada.

O baseline treina em todo o treino da divisao (sem validacao), entao usa um pouco
MAIS dados que os modelos neurais. Isso favorece o baseline: se a CNN/LSTM ainda
ganhar, o ganho nao e artefato.

Exemplos:
    python train_baseline.py --data landmarks_unificado.npz --config modelo_libras_unificado_config.json
    python train_baseline.py --data landmarks_minds_libras_trimmed_interpolated.npz --face_norm --trajectory --model knn
"""
import argparse
import json
import os
import re

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, StratifiedKFold,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

STATIC_SPLIT_KEYS = ("split", "splits", "split_names")


def parse_class_filter(value):
    if value is None:
        return None
    if value.startswith("@"):
        with open(value[1:], encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    return {c.strip() for c in value.split(",") if c.strip()}


def build_features(data, config):
    """Maos + rosto + trajetoria concatenados por frame (mesma montagem do treino)."""
    keys = set(data.files)
    X_hands = data["X_hands"]
    parts = [X_hands.reshape(X_hands.shape[0], X_hands.shape[1], -1)]

    if config.get("use_face", True):
        face_key = "X_face_norm" if config.get("face_norm") else "X_face"
        if face_key not in keys:
            raise SystemExit(f"A representacao pede '{face_key}', mas o .npz so tem: {sorted(keys)}")
        X_face = data[face_key]
        parts.append(X_face.reshape(X_face.shape[0], X_face.shape[1], -1))

    if config.get("trajectory"):
        if "X_traj" not in keys:
            raise SystemExit("A representacao pede trajetoria, mas o .npz nao tem 'X_traj'.")
        X_traj = data["X_traj"]
        parts.append(X_traj.reshape(X_traj.shape[0], X_traj.shape[1], -1))

    return np.concatenate(parts, axis=-1).astype(np.float32)


def get_representation(args):
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = json.load(f)
        return {"use_face": cfg.get("use_face", True), "face_norm": cfg.get("face_norm", False),
                "trajectory": cfg.get("trajectory", False)}
    return {"use_face": not args.no_face, "face_norm": args.face_norm, "trajectory": args.trajectory}


def make_classifier(kind, seed, args):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=args.n_estimators, random_state=seed, n_jobs=-1)
    return KNeighborsClassifier(n_neighbors=args.k, n_jobs=-1)


def signer_ids_from_paths(paths, regex):
    pattern = re.compile(regex)
    ids = []
    for p in paths:
        m = pattern.search(str(p))
        ids.append(m.group(1) if m else "?")
    return np.array(ids)


def make_splits(y, args, signer_ids=None, static_split=None):
    idx = np.arange(len(y))
    if static_split is not None:
        names = np.array([str(s).lower() for s in static_split])
        train, test = idx[names == "train"], idx[names == "test"]
        if len(train) and len(test):
            print(f"Divisao train/test do proprio .npz: {len(train)} treino / {len(test)} teste.")
            return [(train, test)]
        print("A chave de split do .npz nao tem 'train' e 'test' - usando divisao estratificada.")

    if args.by_signer:
        if signer_ids is None:
            raise SystemExit("--by_signer precisa da chave 'paths' no .npz.")
        n_signers = len(set(signer_ids.tolist()))
        print(f"Sinalizadores encontrados ({n_signers}): {sorted(set(signer_ids.tolist()))}")
        if args.folds >= 2:
            return list(GroupKFold(n_splits=min(args.folds, n_signers)).split(idx, y, groups=signer_ids))
        return list(GroupShuffleSplit(n_splits=1, test_size=args.test_size,
                                      random_state=args.split_seed).split(idx, y, groups=signer_ids))

    if args.folds >= 2:
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.split_seed)
        return list(skf.split(idx, y))
    train, test = train_test_split(idx, test_size=args.test_size, random_state=args.split_seed, stratify=y)
    return [(train, test)]


def main():
    parser = argparse.ArgumentParser(description="Baseline Random Forest / k-NN sobre os mesmos vetores dos modelos neurais.")
    parser.add_argument("--data", default="landmarks_unificado.npz", help="Arquivo .npz (video ou alfabeto)")
    parser.add_argument("--model", choices=["rf", "knn"], default="rf", help="Random Forest ou k-NN")
    parser.add_argument("--experiment", default=None, help="Nome do experimento (padrao: baseline_<model>)")
    parser.add_argument("--results_dir", default="resultados")
    parser.add_argument("--n_estimators", type=int, default=300, help="Arvores do Random Forest")
    parser.add_argument("--k", type=int, default=5, help="Vizinhos do k-NN")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Sementes (so o Random Forest usa; o k-NN e deterministico)")
    parser.add_argument("--test_size", type=float, default=0.375, help="Mesmo valor do treino da LSTM")
    parser.add_argument("--split_seed", type=int, default=42, help="Mesmo random_state do treino da LSTM")
    parser.add_argument("--folds", type=int, default=0, help="0 = divisao unica; K >= 2 = validacao cruzada")
    parser.add_argument("--by_signer", action="store_true", help="So video com 'paths': divide por sinalizador (GroupShuffleSplit/GroupKFold)")
    parser.add_argument("--signer_regex", default=r"Sinalizador(\d+)")
    parser.add_argument("--config", default=None, help="_config.json de um modelo treinado: usa a mesma representacao dele")
    parser.add_argument("--no_face", action="store_true", help="Video: ignora o rosto")
    parser.add_argument("--face_norm", action="store_true", help="Video: usa X_face_norm em vez de X_face")
    parser.add_argument("--trajectory", action="store_true", help="Video: acrescenta X_traj")
    parser.add_argument("--classes", default=None, help="Filtra o vocabulario (lista com virgulas ou @arquivo.txt)")
    args = parser.parse_args()

    experiment = args.experiment or f"baseline_{args.model}"
    data = np.load(args.data, allow_pickle=True)
    print(f"Chaves do .npz: {data.files}")

    is_video = "X_hands" in data.files
    static_split = None
    if is_video:
        rep = get_representation(args)
        print(f"Representacao: {rep}")
        X = build_features(data, rep)
    else:
        if "X" not in data.files:
            raise SystemExit("O .npz nao tem 'X_hands' (video) nem 'X' (alfabeto).")
        rep = None
        X = data["X"].astype(np.float32)
        for key in STATIC_SPLIT_KEYS:
            if key in data.files:
                static_split = data[key]
                break

    y_raw = data["y"].astype(str)
    paths = data["paths"] if "paths" in data.files else None
    origem = data["origem"] if "origem" in data.files else None

    class_filter = parse_class_filter(args.classes)
    if class_filter is not None:
        mask = np.array([label in class_filter for label in y_raw])
        X, y_raw = X[mask], y_raw[mask]
        paths = paths[mask] if paths is not None else None
        origem = origem[mask] if origem is not None else None
        static_split = static_split[mask] if static_split is not None else None
        print(f"Filtrando para {len(class_filter)} classes pedidas: {len(y_raw)} amostras restantes.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    X_flat = X.reshape(len(X), -1)
    print(f"Vetor achatado: {X_flat.shape[1]} valores por amostra | {len(y)} amostras | {len(encoder.classes_)} classes.")

    signer_ids = signer_ids_from_paths(paths, args.signer_regex) if (args.by_signer and paths is not None) else None
    splits = make_splits(y, args, signer_ids, static_split)

    seeds = args.seeds if args.model == "rf" else [0]
    runs = []
    first_seed_true, first_seed_pred, first_seed_idx = [], [], []

    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"\n=== Divisao {fold + 1}/{len(splits)}: {len(train_idx)} treino / {len(test_idx)} teste ===")
        for s_i, seed in enumerate(seeds):
            clf = make_classifier(args.model, seed, args)
            clf.fit(X_flat[train_idx], y[train_idx])
            y_pred = clf.predict(X_flat[test_idx])
            acc = accuracy_score(y[test_idx], y_pred)
            f1 = f1_score(y[test_idx], y_pred, average="macro", zero_division=0)
            run = {"divisao": fold + 1, "seed": seed, "acuracia": float(acc), "f1_macro": float(f1)}
            if origem is not None:
                run["acuracia_por_origem"] = {
                    str(o): float(np.mean(y_pred[origem[test_idx] == o] == y[test_idx][origem[test_idx] == o]))
                    for o in np.unique(origem[test_idx])}
            runs.append(run)
            print(f"  seed {seed}: acuracia {acc:.4f} | F1 macro {f1:.4f}")
            if s_i == 0:
                first_seed_true.append(y[test_idx])
                first_seed_pred.append(y_pred)
                first_seed_idx.append(test_idx)

    accs = np.array([r["acuracia"] for r in runs])
    f1s = np.array([r["f1_macro"] for r in runs])
    print(f"\nRESUMO ({args.model}, {len(runs)} execucoes): "
          f"acuracia {accs.mean():.4f} +- {accs.std():.4f} | F1 macro {f1s.mean():.4f} +- {f1s.std():.4f}")

    if origem is not None:
        print("Acuracia por origem (media das execucoes):")
        for o in sorted({k for r in runs for k in r["acuracia_por_origem"]}):
            vals = [r["acuracia_por_origem"][o] for r in runs if o in r["acuracia_por_origem"]]
            print(f"  {o}: {np.mean(vals):.4f}")

    yt, yp = np.concatenate(first_seed_true), np.concatenate(first_seed_pred)
    present = np.unique(np.concatenate([yt, yp]))
    print("\nRelatorio por classe (primeira semente):")
    print(classification_report(yt, yp, labels=present, target_names=encoder.classes_[present], zero_division=0))

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, f"{experiment}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"experimento": experiment, "argumentos": vars(args), "representacao": rep,
                   "classes": encoder.classes_.tolist(), "execucoes": runs,
                   "acuracia_media": float(accs.mean()), "acuracia_desvio": float(accs.std()),
                   "f1_macro_medio": float(f1s.mean()), "f1_macro_desvio": float(f1s.std())},
                  f, ensure_ascii=False, indent=2)
    print(f"Resultados salvos em {out_path}")


if __name__ == "__main__":
    main()