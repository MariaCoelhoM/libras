import argparse
import json
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


def load_features(data_path, use_face=True):
    data = np.load(data_path, allow_pickle=True)
    y_raw = data["y"]

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

    return X, y_raw

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
    parser.add_argument("--test_size", type=float, default=0.3, help="Fracao para teste")
    parser.add_argument("--val_split", type=float, default=0.15, help="Fracao (do treino) para validacao. Use 0 se houver poucos exemplos por classe.")
    parser.add_argument("--no_face", action="store_true", help="Ignora os landmarks faciais mesmo se presentes no .npz - treina so com as maos (util para comparar com/sem rosto).")
    args = parser.parse_args()

    X, y_raw = load_features(args.data, use_face=not args.no_face)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    callbacks = []
    validation_data = None

    if args.val_split > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=args.val_split, stratify=y_train, random_state=42
        )
        validation_data = (X_val, y_val)
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        )
        print(f"Classes: {num_classes}")
        print(f"Treino: {len(X_train)} | Validacao: {len(X_val)} | Teste: {len(X_test)}")
    else:
        print(f"Classes: {num_classes}")
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

    print(f"Modelo salvo em {args.output}")
    print(f"Labels salvas em {labels_path}")


if __name__ == "__main__":
    main()
