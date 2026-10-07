import argparse
import csv
from collections import Counter

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


def load_class_filter(classes_arg):
    """--classes pode ser uma lista separada por virgula, ou @arquivo.txt (uma classe por linha)."""
    if classes_arg is None:
        return None
    if classes_arg.startswith("@"):
        with open(classes_arg[1:], encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    return {c.strip() for c in classes_arg.split(",") if c.strip()}


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


def build_model(num_points, num_channels, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(num_points, num_channels)),

        tf.keras.layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),

        tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
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
    parser.add_argument("--data", required=True, help="Arquivo .npz gerado por extract_landmarks_imagem.py")
    parser.add_argument("--output", default="modelo_alfabeto.keras", help="Caminho para salvar o modelo treinado")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_split", type=float, default=0.15, help="Fracao do treino original usada para validacao (so entra em efeito quando o .npz tem o split train/test original).")
    parser.add_argument("--classes", default=None, help="Filtra o vocabulario: lista separada por virgula, ou @arquivo.txt com uma classe por linha.")
    parser.add_argument("--counts_output", default=None, help="Caminho de um .csv para salvar a contagem de amostras por classe e por divisao.")
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    X, y_raw = data["X"], data["y"]
    split_names = data["split"] if "split" in data else None

    class_filter = load_class_filter(args.classes)
    if class_filter is not None:
        mask = np.array([label in class_filter for label in y_raw])
        X, y_raw = X[mask], y_raw[mask]
        if split_names is not None:
            split_names = split_names[mask]
        print(f"Filtrando para {len(class_filter)} classes pedidas: {len(y_raw)} amostras restantes.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)

    counts_by_split = {}

    tem_split_original = split_names is not None and {"train", "test"}.issubset(set(split_names.tolist()))

    if tem_split_original:
        print("Usando a divisao original train/test do dataset (nao re-sorteando amostras).")
        idx_train_full = np.where(split_names == "train")[0]
        idx_test = np.where(split_names == "test")[0]

        idx_sub_train, idx_sub_val = train_test_split(
            idx_train_full, test_size=args.val_split, stratify=y[idx_train_full], random_state=42
        )

        X_train, y_train = X[idx_sub_train], y[idx_sub_train]
        X_val, y_val = X[idx_sub_val], y[idx_sub_val]
        X_test, y_test = X[idx_test], y[idx_test]

        log_counts("treino", y_raw[idx_sub_train], counts_by_split)
        log_counts("validacao", y_raw[idx_sub_val], counts_by_split)
        log_counts("teste", y_raw[idx_test], counts_by_split)
    else:
        print("Aviso: o .npz nao tem a divisao original train/test (ou esta incompleta) - "
              "fazendo split aleatorio estratificado 70/15/15. Para usar a divisao original do "
              "dataset, re-extraia os landmarks com o extract_landmarks_imagem.py atualizado.")
        idx_all = np.arange(len(y))
        idx_train, idx_temp = train_test_split(idx_all, test_size=0.3, stratify=y, random_state=42)
        idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, stratify=y[idx_temp], random_state=42)

        X_train, y_train = X[idx_train], y[idx_train]
        X_val, y_val = X[idx_val], y[idx_val]
        X_test, y_test = X[idx_test], y[idx_test]

        log_counts("treino", y_raw[idx_train], counts_by_split)
        log_counts("validacao", y_raw[idx_val], counts_by_split)
        log_counts("teste", y_raw[idx_test], counts_by_split)

    print(f"\nTreino: {len(X_train)} | Validacao: {len(X_val)} | Teste: {len(X_test)}")
    print(f"Classes: {list(encoder.classes_)}")

    model = build_model(num_points=X.shape[1], num_channels=X.shape[2], num_classes=num_classes)
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop],
    )

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nAcuracia no teste: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nMatriz de confusao:")
    print(confusion_matrix(y_test, y_pred))
    print("\nRelatorio de classificacao:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    if args.counts_output:
        save_counts_csv(args.counts_output, counts_by_split)

    model.save(args.output)
    print(f"Modelo salvo em {args.output}")


if __name__ == "__main__":
    main()
    