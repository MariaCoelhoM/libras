"""Matriz de confusao do modelo unificado de palavras (MINDS-Libras + V-Librasil).

Le o landmarks_unificado.npz, monta as features exatamente como o treino
(maos + rosto normalizado + trajetoria, conforme o _config.json do modelo),
reproduz a divisao treino/teste (stratify + random_state) e avalia so o teste.

Uso:
    python matriz.py --model modelo_libras_unificado.keras --data landmarks_unificado.npz

IMPORTANTE: o split so e igual ao do treino se o treino usou os mesmos valores de
--test_size/--seed (padrao do script de treino: 0.375 e 42). Confira: a acuracia
impressa aqui deve bater com a "Acuracia no teste" do log do treino.
"""
import argparse
import json
import os

import numpy as np


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_features(data, config):
    """Concatena maos + rosto + trajetoria por frame, de acordo com o config do modelo."""
    keys = set(data.files)
    X_hands = data["X_hands"]
    parts = [X_hands.reshape(X_hands.shape[0], X_hands.shape[1], -1)]

    if config.get("use_face", True):
        face_key = "X_face_norm" if config.get("face_norm", True) else "X_face"
        if face_key not in keys:
            raise SystemExit(f"O config pede '{face_key}', mas o .npz so tem: {sorted(keys)}")
        X_face = data[face_key]
        parts.append(X_face.reshape(X_face.shape[0], X_face.shape[1], -1))

    if config.get("trajectory", False):
        if "X_traj" not in keys:
            raise SystemExit("O config pede trajetoria, mas o .npz nao tem 'X_traj'.")
        X_traj = data["X_traj"]
        parts.append(X_traj.reshape(X_traj.shape[0], X_traj.shape[1], -1))

    return np.concatenate(parts, axis=-1).astype(np.float32)


def encode_labels(y_raw, labels):
    """Usa a ordem do _labels.json (a mesma do LabelEncoder do treino)."""
    index = {name: i for i, name in enumerate(labels)}
    desconhecidas = sorted({str(v) for v in y_raw if str(v) not in index})
    if desconhecidas:
        raise SystemExit(f"Classes no .npz que nao estao no _labels.json: {desconhecidas}")
    return np.array([index[str(v)] for v in y_raw])


def plot_confusion(cm, classes, output, normalize=False):
    import matplotlib
    matplotlib.use("Agg")  # sem janela (Codespace)
    import matplotlib.pyplot as plt

    n = len(classes)
    if normalize:
        shown = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    else:
        shown = cm
    fig, ax = plt.subplots(figsize=(max(11, n * 0.5), max(10, n * 0.46)))
    ax.imshow(shown, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusao" + (" (normalizada por linha)" if normalize else " (contagens)"))
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                text = f"{shown[i, j]:.2f}" if normalize else str(cm[i, j])
                ax.text(j, i, text, ha="center", va="center", fontsize=7,
                        color="white" if shown[i, j] > shown.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(output, dpi=150)


def main():
    parser = argparse.ArgumentParser(description="Matriz de confusao do modelo unificado no conjunto de teste.")
    parser.add_argument("--model", default="modelo_libras_unificado.keras")
    parser.add_argument("--data", default="landmarks_unificado.npz", help=".npz unificado usado no treino")
    parser.add_argument("--test_size", type=float, default=0.375, help="Mesmo valor do treino")
    parser.add_argument("--seed", type=int, default=42, help="Mesmo random_state do treino")
    parser.add_argument("--normalize", action="store_true", help="Matriz normalizada por linha (recomendado com classes desbalanceadas)")
    parser.add_argument("--output", default="matriz_confusao.png")
    args = parser.parse_args()

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix

    base = os.path.splitext(args.model)[0]
    labels = load_json(base + "_labels.json")
    config_path = base + "_config.json"
    config = load_json(config_path) if os.path.exists(config_path) else {}
    print(f"Representacao: {config}")

    data = np.load(args.data, allow_pickle=True)
    X = build_features(data, config)
    y = encode_labels(data["y"], labels)
    origem = data["origem"] if "origem" in data.files else None

    esperado = (config.get("timesteps"), config.get("n_features"))
    if None not in esperado and tuple(X.shape[1:]) != esperado:
        raise SystemExit(f"As features montadas tem shape {X.shape[1:]}, mas o modelo espera {esperado} "
                         f"(timesteps, n_features) segundo {config_path}.")
    print(f"Amostras: {len(X)} | features por frame: {X.shape[-1]} | classes: {len(labels)}")

    # Mesma divisao do treino: stratify=y, random_state fixo
    _, test_idx = train_test_split(np.arange(len(X)), test_size=args.test_size,
                                   random_state=args.seed, stratify=y)
    X_test, y_test = X[test_idx], y[test_idx]
    print(f"Teste: {len(test_idx)} amostras")

    import tensorflow as tf
    model = tf.keras.models.load_model(args.model)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(f"\nAcuracia no teste: {np.mean(y_pred == y_test):.4f}  (compare com o log do treino)")

    if origem is not None:
        print("\nAcuracia por origem (no teste):")
        for o in np.unique(origem[test_idx]):
            m = origem[test_idx] == o
            print(f"  {o}: {np.mean(y_pred[m] == y_test[m]):.4f}  ({int(m.sum())} amostras)")

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(labels)))

    print("\nAcerto por classe (recall), do pior ao melhor:")
    rows = []
    for i, name in enumerate(labels):
        total = int(cm[i].sum())
        if total:
            rows.append((cm[i, i] / total, name, int(cm[i, i]), total))
    for recall, name, ok, total in sorted(rows):
        print(f"  {name}: {recall:.2f} ({ok}/{total})")

    errors = [(cm[i, j], labels[i], labels[j])
              for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j] > 0]
    print("\nConfusoes mais frequentes (real -> predito):")
    for count, real, pred in sorted(errors, reverse=True)[:10]:
        print(f"  {real} -> {pred}: {count}")

    plot_confusion(cm, labels, args.output, normalize=args.normalize)
    print(f"\nMatriz de confusao salva em {args.output}")


if __name__ == "__main__":
    main()