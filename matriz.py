import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")  # sem janela (Codespace)
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

from libras.training_utils import load_class_filter, extract_signer_ids
from train_lstm_hands_face import load_data, build_points, to_features


def load_config(model_path, args):
    """Representacao usada no treino: vem do _config.json salvo ao lado do modelo.
    Modelos sem o arquivo (ex. o v3) usam as flags; sem flags, vale o padrao do treino
    (rosto + trajetoria + velocidade)."""
    config_path = os.path.splitext(model_path)[0] + "_config.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            print(f"Representacao lida de {config_path}")
            return json.load(f)
    print(f"{config_path} nao encontrado - usando as flags (padrao: rosto + trajetoria + velocidade).")
    return {"use_face": not args.no_face, "use_traj": not args.no_traj, "use_velocity": not args.no_velocity}


def plot_confusion(cm, classes, output):
    n = len(classes)
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(output, dpi=150)


def main():
    parser = argparse.ArgumentParser(description="Matriz de confusao de UM modelo salvo, no conjunto de teste da divisao unica do treino.")
    parser.add_argument("--data", required=True, help="Mesmo .npz usado no treino")
    parser.add_argument("--model", required=True, help="Arquivo .keras salvo pelo treino")
    parser.add_argument("--test_size", type=float, default=0.3, help="Mesmo valor usado no treino")
    parser.add_argument("--classes", default=None, help="Mesmo filtro de classes usado no treino")
    parser.add_argument("--no_face", action="store_true", help="So vale para modelos sem _config.json: use se treinou com --no_face")
    parser.add_argument("--no_traj", action="store_true", help="So vale para modelos sem _config.json: use se treinou com --no_traj")
    parser.add_argument("--no_velocity", action="store_true", help="So vale para modelos sem _config.json: use se treinou com --no_velocity")
    parser.add_argument("--signer_regex", default=r"Sinalizador(\d+)", help="Mesmo regex usado no treino")
    parser.add_argument("--output", default="matriz_confusao.png", help="PNG de saida")
    args = parser.parse_args()

    config = load_config(args.model, args)

    X_hands, X_face, X_traj, y_raw, paths = load_data(args.data)
    points = build_points(X_hands, X_face, X_traj, config["use_face"], config["use_traj"])

    class_filter = load_class_filter(args.classes)
    if class_filter is not None:
        mask = np.array([label in class_filter for label in y_raw])
        points, y_raw = points[mask], y_raw[mask]
        if paths is not None:
            paths = paths[mask]

    if paths is None:
        raise SystemExit("O .npz nao tem 'paths' - nao da pra reproduzir o split por sinalizador do treino.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    signer_ids = extract_signer_ids(paths, re.compile(args.signer_regex))

    # Mesma divisao por sinalizador do treino (mesmo test_size e random_state)
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
    _, test_idx = next(splitter.split(points, y, groups=signer_ids))
    print(f"Sinalizadores no teste: {sorted(set(signer_ids[test_idx].tolist()))}")

    X_test = to_features(points[test_idx], config["use_velocity"])
    y_test = y[test_idx]

    model = tf.keras.models.load_model(args.model)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(f"Acuracia no teste: {np.mean(y_pred == y_test):.4f}")

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(encoder.classes_)))

    errors = [(cm[i, j], encoder.classes_[i], encoder.classes_[j])
              for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j] > 0]
    print("\nConfusoes mais frequentes (real -> predito):")
    for count, real, pred in sorted(errors, reverse=True)[:10]:
        print(f"  {real} -> {pred}: {count}")

    plot_confusion(cm, encoder.classes_, args.output)
    print(f"\nMatriz de confusao salva em {args.output}")


if __name__ == "__main__":
    main()