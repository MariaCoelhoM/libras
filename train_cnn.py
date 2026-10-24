import argparse
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from libras.experiments import ExperimentLog, set_seed, static_splits, static_train_val
from libras.features import load_static_features
from libras.training_utils import load_class_filter, log_counts, save_counts_csv


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
    parser.add_argument("--experiment", default=None, help="Nome do experimento: os resultados vao para <results_dir>/<experiment>/. Se omitido, usa o nome de --output.")
    parser.add_argument("--results_dir", default="resultados", help="Pasta onde gravar os resultados")
    parser.add_argument("--output", default=None, help="Se passado, salva o modelo da PRIMEIRA semente neste .keras (o app espera modelo_alfabeto.keras), com _labels.json.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Sementes do TensorFlow; o treino e repetido uma vez por semente para estimar a variancia. A divisao dos dados e a original do dataset (fixa).")
    parser.add_argument("--deterministic", action="store_true", help="Pede operacoes deterministicas ao TensorFlow (mesma semente, mesmo resultado exato; mais lento).")
    parser.add_argument("--val_split", type=float, default=0.15, help="Fracao do treino original usada para validacao/early stopping")
    parser.add_argument("--classes", default=None, help="Filtra o vocabulario: lista separada por virgula, ou @arquivo.txt com uma classe por linha.")
    parser.add_argument("--counts_output", default=None, help="(Opcional) copia da contagem de amostras neste .csv. A contagem ja vai para <results_dir>/<experiment>/contagem_amostras.csv.")
    args = parser.parse_args()

    experiment = args.experiment or (os.path.splitext(os.path.basename(args.output))[0] if args.output else None)
    if experiment is None:
        raise SystemExit("Passe --experiment NOME (ou --output, de onde o nome e tirado).")

    X, y_raw, split_names = load_static_features(args.data)

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

    (train_idx, test_idx), = static_splits(y, split_names)
    fit_idx, val_idx = static_train_val(train_idx, y, args.val_split)

    counts_by_split = {}
    log_counts("treino", y_raw[fit_idx], counts_by_split)
    if val_idx is not None:
        log_counts("validacao", y_raw[val_idx], counts_by_split)
    log_counts("teste", y_raw[test_idx], counts_by_split)
    print(f"\nTreino: {len(fit_idx)} | Validacao: {0 if val_idx is None else len(val_idx)} | Teste: {len(test_idx)}")
    print(f"Classes: {list(encoder.classes_)}")

    log = ExperimentLog(experiment, encoder.classes_)

    for run_idx, seed in enumerate(args.seeds):
        set_seed(seed, args.deterministic)
        model = build_model(num_points=X.shape[1], num_channels=X.shape[2], num_classes=num_classes)
        if run_idx == 0:
            model.summary()

        callbacks, validation_data = [], None
        if val_idx is not None:
            validation_data = (X[val_idx], y[val_idx])
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True))

        history = model.fit(
            X[fit_idx], y[fit_idx],
            validation_data=validation_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        y_pred = np.argmax(model.predict(X[test_idx], verbose=0), axis=1)
        log.add_run(0, seed, y[test_idx], y_pred,
                    epocas_treinadas=len(history.history["loss"]), n_treino=int(len(fit_idx)))

        if run_idx == 0 and args.output:
            model.save(args.output)
            with open(os.path.splitext(args.output)[0] + "_labels.json", "w") as f:
                json.dump(list(encoder.classes_), f, ensure_ascii=False)
            print(f"Modelo da primeira semente salvo em {args.output}")
        tf.keras.backend.clear_session()

    log.print_report()
    log.save(args.results_dir, {**vars(args), "divisao": "original train/test do dataset"}, counts_by_split)
    if args.counts_output:
        save_counts_csv(args.counts_output, counts_by_split)


if __name__ == "__main__":
    main()