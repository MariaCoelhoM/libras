"""Baseline simples (Random Forest ou k-NN) para medir o ganho da CNN e da LSTM.

Usa os MESMOS vetores dos modelos neurais, so que achatados, e a MESMA divisao:
  - .npz de alfabeto (extract_landmarks_imagem.py): vetor de 21 x 3 = 63 valores,
    divisao original train/test do dataset;
  - .npz de video (extract_landmarks_video_*.py): sequencia (T x features) achatada
    em um vetor so, divisao por sinalizador (--folds), com as mesmas opcoes de
    representacao do train_lstm_hands_face.py (--no_face, --face_norm, ...).

O baseline treina em todo o treino da divisao (nao precisa de validacao), entao
usa um pouco MAIS dados de treino que os modelos neurais, que separam uma parte
para o early stopping. Isso favorece o baseline: se a CNN/LSTM ainda ganhar,
o ganho nao e artefato.
"""

import argparse
import re

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from libras.experiments import ExperimentLog, set_seed, signer_splits, static_splits
from libras.features import load_sequence_features, load_static_features
from libras.training_utils import load_class_filter, extract_signer_ids, log_counts, save_counts_csv


def make_classifier(kind, seed, args):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=args.n_estimators, random_state=seed, n_jobs=-1)
    return KNeighborsClassifier(n_neighbors=args.k, n_jobs=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Arquivo .npz (alfabeto ou video)")
    parser.add_argument("--model", choices=["rf", "knn"], default="rf", help="Random Forest ou k-NN")
    parser.add_argument("--experiment", default=None, help="Nome do experimento (padrao: baseline_<model>)")
    parser.add_argument("--results_dir", default="resultados")
    parser.add_argument("--n_estimators", type=int, default=300, help="Arvores do Random Forest")
    parser.add_argument("--k", type=int, default=5, help="Vizinhos do k-NN")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Sementes (so o Random Forest usa; o k-NN e deterministico e roda uma vez).")
    parser.add_argument("--folds", type=int, default=0, help="So para video: 0 = divisao unica por sinalizador; K >= 2 = validacao cruzada por sinalizador.")
    parser.add_argument("--leave_one_signer_out", action="store_true", help="So para video: deixa um sinalizador de fora por rodada.")
    parser.add_argument("--test_size", type=float, default=0.3, help="So para video, com --folds 0")
    parser.add_argument("--no_face", action="store_true", help="So para video: ignora o rosto")
    parser.add_argument("--face_norm", action="store_true", help="So para video: face normalizada")
    parser.add_argument("--trajectory", action="store_true", help="So para video: acrescenta a trajetoria do punho (X_traj)")
    parser.add_argument("--velocity", action="store_true", help="So para video: acrescenta a velocidade do punho")
    parser.add_argument("--classes", default=None, help="Filtra o vocabulario (lista com virgulas ou @arquivo.txt)")
    parser.add_argument("--signer_regex", default=r"Sinalizador(\d+)", help="So para video")
    args = parser.parse_args()

    experiment = args.experiment or f"baseline_{args.model}"

    data_keys = np.load(args.data, allow_pickle=True)
    is_video = "X_hands" in data_keys.files or data_keys["X"].ndim == 5

    if is_video:
        config = {"use_face": not args.no_face, "face_norm": args.face_norm,
                  "trajectory": args.trajectory, "velocity": args.velocity}
        X, y_raw, paths, config = load_sequence_features(args.data, config)
        aux = paths
        split_names = None
    else:
        X, y_raw, split_names = load_static_features(args.data)
        config, paths, aux = None, None, split_names

    class_filter = load_class_filter(args.classes)
    if class_filter is not None:
        mask = np.array([label in class_filter for label in y_raw])
        X, y_raw = X[mask], y_raw[mask]
        if aux is not None:
            aux = aux[mask]
        print(f"Filtrando para {len(class_filter)} classes pedidas: {len(y_raw)} amostras restantes.")
    if is_video:
        paths = aux
    else:
        split_names = aux

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_flat = X.reshape(len(X), -1)
    print(f"Vetor achatado: {X_flat.shape[1]} valores por amostra ({'video' if is_video else 'alfabeto'}).")

    if is_video:
        if paths is None:
            raise SystemExit("O .npz nao tem 'paths' - nao da pra dividir por sinalizador.")
        signer_ids = extract_signer_ids(paths, re.compile(args.signer_regex))
        n_signers = len(set(signer_ids.tolist()))
        print(f"Sinalizadores encontrados ({n_signers}): {sorted(set(signer_ids.tolist()))}")
        folds = n_signers if args.leave_one_signer_out else args.folds
        splits = signer_splits(signer_ids, folds, args.test_size)
    else:
        signer_ids = None
        splits = static_splits(y, split_names)

    seeds = args.seeds if args.model == "rf" else [0]
    if args.model == "knn" and len(args.seeds) > 1:
        print("k-NN e deterministico: uma execucao por dobra (as sementes extras seriam identicas).")

    log = ExperimentLog(experiment, encoder.classes_)
    counts_by_split = {}

    for fold, (train_idx, test_idx) in enumerate(splits):
        tag = f"dobra{fold + 1}"
        if signer_ids is not None:
            print(f"\n=== Dobra {fold + 1}/{len(splits)} - sinalizador(es) no teste: "
                  f"{sorted(set(signer_ids[test_idx].tolist()))} ===")
        log_counts(f"{tag}_treino", y_raw[train_idx], counts_by_split)
        log_counts(f"{tag}_teste", y_raw[test_idx], counts_by_split)

        for seed in seeds:
            set_seed(seed, use_tensorflow=False)
            clf = make_classifier(args.model, seed, args)
            clf.fit(X_flat[train_idx], y[train_idx])
            y_pred = clf.predict(X_flat[test_idx])
            extra = {"n_treino": int(len(train_idx))}
            if signer_ids is not None:
                extra["sinalizadores_teste"] = sorted(set(signer_ids[test_idx].tolist()))
            log.add_run(fold, seed, y[test_idx], y_pred, **extra)

    log.print_report()
    log.save(args.results_dir, {**vars(args), "tipo_de_dado": "video" if is_video else "alfabeto",
                                "representacao": config}, counts_by_split)


if __name__ == "__main__":
    main()
