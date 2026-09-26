import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


def load_features(data_path, use_face=True):
    data = np.load(data_path, allow_pickle=True)
    y_raw = data["y"]

    X_hands_raw = data["X_hands"]
    N, T = X_hands_raw.shape[0], X_hands_raw.shape[1]
    X_hands = X_hands_raw.reshape(N, T, -1)

    if use_face and "X_face" in data:
        X_face_raw = data["X_face"]
        X_face = X_face_raw.reshape(N, T, -1)
        X = np.concatenate([X_hands, X_face], axis=-1)
    else:
        X = X_hands

    return X, y_raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Mesmo .npz usado no treino")
    parser.add_argument("--model", required=True, help="Arquivo .keras salvo pelo treino")
    parser.add_argument("--test_size", type=float, default=0.3, help="Mesmo valor usado no treino")
    parser.add_argument("--no_face", action="store_true", help="Use se treinou com --no_face")
    args = parser.parse_args()

    X, y_raw = load_features(args.data, use_face=not args.no_face)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    # Mesmo split do treino.py: reproduz o mesmo X_test/y_test
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    model = tf.keras.models.load_model(args.model)
    y_pred = np.argmax(model.predict(X_test), axis=1)

    classes = encoder.classes_
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusao")
    plt.tight_layout()
    plt.savefig("matriz_confusao.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()