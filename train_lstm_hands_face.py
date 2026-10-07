import argparse
import csv
import json
import os
import re
from collections import Counter

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


def load_features(data_path, use_face=True):
    data = np.load(data_path, allow_pickle=True)
    y_raw = data["y"]
    paths = data["paths"] if "paths" in data else None

    if "X_hands" in data:
        X_hands_raw = data["X_hands"]  # (N, T, 2, 21, 3)
        N, T = X_hands_raw.shape[0], X_hands_raw.shape[1]
        X_hands = X_hands_raw.reshape(N, T, -1)  # (N, T, 126)

        if use_face and "X_face" in data:
            X_face_raw = data["X_face"]  # (N, T, P, 3)
            X_face = X_face_raw.reshape(N, T, -1)  # (N, T, P*3)
            X = np.concatenate([X_hands, X_face], axis=-1)
            print(f"Usando maos + rosto: {X_hands.shape[-1]} + {X_face.shape[-1]} = {X.shape[-1]} features por frame")
        else:
            X = X_hands
            if not use_face:
                print(f"Usando so maos (--no_face): {X.shape[-1]} features por frame")
            else:
                print(f"'X_face' nao encontrado no .npz - usando so maos: {X.shape[-1]} features por frame")
    elif "X" in data:
        X_raw = data["X"]
        N, T = X_raw.shape[0], X_raw.shape[1]
        X = X_raw.reshape(N, T, -1)
        print(f"Formato antigo detectado (so maos): {X.shape[-1]} features por frame")
    else:
        raise KeyError(f"{data_path} nao tem 'X_hands' nem 'X' - formato de .npz nao reconhecido.")

    return X, y_raw, paths


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
    ids = np.array(ids)
    if "desconhecido" in ids:
        n = int((ids == "desconhecido").sum())
        print(f"Aviso: {n} arquivo(s) sem sinalizador identificavel pelo regex - "
              f"tratados como um grupo unico 'desconhecido'. Ajuste --signer_regex se isso nao fizer sentido.")
    return ids


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


def build_model(timesteps, num_features, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(timesteps, num_features)),
        tf.keras.layers.Masking(mask_value=0.0),

        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Arquivo .npz gerado por extract_landmarks_video.py")
    parser.add_argument("--output", default="modelo_palavras.keras", help="Caminho para salvar o modelo treinado")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_size", type=float, default=0.3, help="Fracao de sinalizadores para teste")
    parser.add_argument("--val_split", type=float, default=0.15, help="Fracao (dos sinalizadores de treino) para validacao. Use 0 se houver poucos sinalizadores.")
    parser.add_argument("--no_face", action="store_true", help="Ignora os landmarks faciais mesmo se presentes no .npz - treina so com as maos (util para comparar com/sem rosto).")
    parser.add_argument("--classes", default=None, help="Filtra o vocabulario: lista separada por virgula (ex. 'Aluno,Filho,Medo') ou @arquivo.txt com uma classe por linha.")
    parser.add_argument("--signer_regex", default=r"Sinalizador(\d+)", help="Regex com 1 grupo de captura para extrair o ID do sinalizador do nome do arquivo em 'paths'. Padrao serve para MINDS-Libras; ajuste para outros datasets (ex. V-Librasil).")
    parser.add_argument("--leave_one_signer_out", action="store_true", help="Em vez de um unico split treino/teste, faz validacao cruzada deixando um sinalizador de fora por rodada (GroupKFold). Util para datasets com poucos sinalizadores (ex. V-Librasil).")
    parser.add_argument("--counts_output", default=None, help="Caminho de um .csv para salvar a contagem de amostras por classe e por divisao.")
    args = parser.parse_args()

    X, y_raw, paths = load_features(args.data, use_face=not args.no_face)

    class_filter = load_class_filter(args.classes)
    if class_filter is not None:
        mask = np.array([label in class_filter for label in y_raw])
        X, y_raw = X[mask], y_raw[mask]
        if paths is not None:
            paths = paths[mask]
        print(f"Filtrando para {len(class_filter)} classes pedidas: {len(y_raw)} amostras restantes "
              f"({len(set(y_raw.tolist()))} classes de fato encontradas nos dados).")

    if paths is None:
        raise SystemExit(
            "O .npz nao tem 'paths' - nao da pra identificar o sinalizador pra fazer o split por grupo. "
            "Re-extraia os landmarks com um extrator que salve 'paths' (os extratores de video ja salvam)."
        )

    signer_regex = re.compile(args.signer_regex)
    signer_ids = extract_signer_ids(paths, signer_regex)
    print(f"Sinalizadores encontrados ({len(set(signer_ids.tolist()))}): {sorted(set(signer_ids.tolist()))}")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)

    counts_by_split = {}

    if args.leave_one_signer_out:
        unique_signers = sorted(set(signer_ids.tolist()))
        n_splits = len(unique_signers)
        gkf = GroupKFold(n_splits=n_splits)
        accs = []

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=signer_ids)):
            left_out = sorted(set(signer_ids[test_idx].tolist()))
            print(f"\n=== Rodada {fold + 1}/{n_splits} - sinalizador(es) de fora: {left_out} ===")

            log_counts(f"rodada{fold + 1}_treino", y_raw[train_idx], counts_by_split)
            log_counts(f"rodada{fold + 1}_teste", y_raw[test_idx], counts_by_split)

            model = build_model(timesteps=X.shape[1], num_features=X.shape[2], num_classes=num_classes)
            model.fit(X[train_idx], y[train_idx], epochs=args.epochs, batch_size=args.batch_size, verbose=0)

            test_loss, test_acc = model.evaluate(X[test_idx], y[test_idx], verbose=0)
            print(f"Acuracia (sinalizador de fora {left_out}): {test_acc:.4f}")
            accs.append(test_acc)

        print(f"\nAcuracia media entre rodadas: {np.mean(accs):.4f} (+/- {np.std(accs):.4f})")

        if args.counts_output:
            save_counts_csv(args.counts_output, counts_by_split)
        return

    # Split por sinalizador: garante que a mesma pessoa nao aparece em treino e teste
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=signer_ids))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    signers_train = signer_ids[train_idx]

    log_counts("teste", y_raw[test_idx], counts_by_split)

    callbacks = []
    validation_data = None

    if args.val_split > 0:
        val_splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_split, random_state=42)
        sub_train_idx, sub_val_idx = next(val_splitter.split(X_train, y_train, groups=signers_train))

        X_val, y_val = X_train[sub_val_idx], y_train[sub_val_idx]
        y_raw_val = y_raw[train_idx][sub_val_idx]
        y_raw_treino = y_raw[train_idx][sub_train_idx]
        X_train, y_train = X_train[sub_train_idx], y_train[sub_train_idx]

        validation_data = (X_val, y_val)
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        )

        log_counts("validacao", y_raw_val, counts_by_split)
        log_counts("treino", y_raw_treino, counts_by_split)
        print(f"\nClasses: {num_classes}")
        print(f"Treino: {len(X_train)} | Validacao: {len(X_val)} | Teste: {len(X_test)}")
    else:
        log_counts("treino", y_raw[train_idx], counts_by_split)
        print(f"\nClasses: {num_classes}")
        print(f"Treino: {len(X_train)} | Teste: {len(X_test)} (sem validacao separada)")

    model = build_model(timesteps=X.shape[1], num_features=X.shape[2], num_classes=num_classes)
    model.summary()

    model.fit(
        X_train, y_train,
        validation_data=validation_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nAcuracia no teste: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nRelatorio de classificacao (resumido, muitas classes):")
    print(classification_report(
        y_test, y_pred, target_names=encoder.classes_, zero_division=0
    ))

    model.save(args.output)

    labels_path = os.path.splitext(args.output)[0] + "_labels.json"
    with open(labels_path, "w") as f:
        json.dump(list(encoder.classes_), f, ensure_ascii=False)

    if args.counts_output:
        save_counts_csv(args.counts_output, counts_by_split)

    print(f"Modelo salvo em {args.output}")
    print(f"Labels salvas em {labels_path}")


if __name__ == "__main__":
    main()
    