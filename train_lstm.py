"""Une os .npz do MINDS e do Nucleo Comum e treina um unico modelo de palavras.

Exemplo:
    python train_unificado.py --minds landmarks_minds.npz --nucleo landmarks_nucleo_comum.npz

Saidas (na pasta atual, por padrao):
    landmarks_unificado.npz              dados unidos (com a coluna "origem")
    modelo_libras_unificado.keras        modelo
    modelo_libras_unificado_labels.json  classes
    modelo_libras_unificado_config.json  representacao (mesmas chaves do app.py)
    modelo_libras_unificado_relatorio.txt  resultados do teste, por classe e por base
"""

import argparse
import json
import os
import unicodedata

import numpy as np

# Chaves exigidas nos dois .npz. A ordem das features e sempre
# mãos, rosto normalizado e trajetória, a mesma dos treinos anteriores
# e a que o assemble_features do app.py espera.
CHAVES_X = ("X_hands", "X_face_norm", "X_traj")
CHAVES = CHAVES_X + ("y",)


def parse_args():
    p = argparse.ArgumentParser(description="Treino unificado MINDS + Nucleo Comum")
    p.add_argument("--minds", required=True, help=".npz do MINDS")
    p.add_argument("--nucleo", required=True, help=".npz do Nucleo Comum")
    p.add_argument("--output", default="modelo_libras_unificado.keras")
    p.add_argument("--salvar_npz", default="landmarks_unificado.npz",
                   help="Onde salvar os dados unidos")
    p.add_argument("--so_unir", action="store_true",
                   help="Apenas une os .npz e para, sem treinar")
    p.add_argument("--timesteps", type=int, default=30,
                   help="Quadros por sinal (o app.py usa 30)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--augment_copies", type=int, default=5)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Forca rodar na CPU")
    return p.parse_args()


def sem_acento(texto):
    return "".join(
        c
        for c in unicodedata.normalize("NFD", str(texto).strip().lower())
        if unicodedata.category(c) != "Mn"
    )


# ---------------- Uniao dos .npz ----------------

def carregar(caminho, nome, timesteps):
    data = np.load(caminho, allow_pickle=True)

    faltando = [k for k in CHAVES if k not in data.files]
    if faltando:
        raise SystemExit(
            f"[{nome}] faltam as chaves {faltando} em {caminho}.\n"
            f"Chaves presentes: {data.files}"
        )

    dados = {k: data[k] for k in CHAVES}

    tamanhos = {k: len(v) for k, v in dados.items()}
    if len(set(tamanhos.values())) != 1:
        raise SystemExit(f"[{nome}] chaves com numeros de amostras diferentes: {tamanhos}")

    # Leva todas as chaves para o mesmo numero de quadros, escolhendo os
    # quadros pelos mesmos indices que o app.py usa (linspace + round).
    for k in CHAVES_X:
        t_atual = dados[k].shape[1]

        if t_atual != timesteps:
            idx = np.linspace(0, t_atual - 1, timesteps).round().astype(int)
            dados[k] = dados[k][:, idx]
            print(f"[{nome}] {k}: {t_atual} -> {timesteps} quadros")

    dados["y"] = np.array([str(v) for v in dados["y"]])

    print(
        f"[{nome}] {len(dados['y'])} amostras, "
        f"{len(set(dados['y']))} classes | "
        + ", ".join(f"{k} {dados[k].shape[1:]}" for k in CHAVES_X)
    )

    return dados


def unir(minds, nucleo):
    for k in CHAVES_X:
        a, b = minds[k].shape[2:], nucleo[k].shape[2:]

        if a != b:
            raise SystemExit(
                f"{k} incompativel: MINDS {a} x Nucleo {b}. "
                "Os dois .npz precisam ter sido extraidos com o mesmo extractor "
                "e as mesmas opcoes (face_norm, trajectory)."
            )

    # Mesma classe com grafia diferente (acento ou caixa) vira uma so.
    canonico = {}
    avisados = set()

    def uniformizar(rotulos):
        saida = []

        for r in rotulos:
            k = sem_acento(r)
            canonico.setdefault(k, r)

            if canonico[k] != r and (r, canonico[k]) not in avisados:
                print(f"Aviso: '{r}' tratado como '{canonico[k]}'")
                avisados.add((r, canonico[k]))

            saida.append(canonico[k])

        return np.array(saida)

    y_minds = uniformizar(minds["y"])
    y_nucleo = uniformizar(nucleo["y"])

    comuns = sorted(str(c) for c in set(y_minds) & set(y_nucleo))
    print(f"Classes em comum entre as bases: {comuns if comuns else 'nenhuma'}")

    unido = {
        k: np.concatenate([minds[k], nucleo[k]], axis=0)
        for k in CHAVES_X
    }
    unido["y"] = np.concatenate([y_minds, y_nucleo])
    unido["origem"] = np.array(
        ["minds"] * len(y_minds) + ["nucleo"] * len(y_nucleo)
    )

    return unido


# ---------------- Data augmentation (a mesma do treino do Nucleo) ----------------

def aug_gaussian_noise(seq, nivel=0.008):
    return seq + np.random.normal(0, nivel, seq.shape)


def aug_spatial_scaling(seq, faixa=(0.92, 1.08)):
    return seq * np.random.uniform(*faixa)


def aug_temporal_resampling(seq):
    T, n = seq.shape
    novo_T = max(5, int(T * np.random.uniform(0.85, 1.15)))

    antigos = np.linspace(0, T - 1, T)
    novos = np.linspace(0, T - 1, novo_T)
    finais = np.linspace(0, novo_T - 1, T)

    saida = np.zeros((T, n))

    for f in range(n):
        reamostrado = np.interp(novos, antigos, seq[:, f])
        saida[:, f] = np.interp(finais, np.arange(novo_T), reamostrado)

    return saida


def aplicar_augmentation(X, y, copias):
    X_lista, y_lista = [X], [y]

    for _ in range(copias):
        novas = []

        for seq in X:
            s = seq.copy()

            if np.random.rand() > 0.3:
                s = aug_gaussian_noise(s)
            if np.random.rand() > 0.3:
                s = aug_spatial_scaling(s)
            if np.random.rand() > 0.4:
                s = aug_temporal_resampling(s)

            novas.append(s)

        X_lista.append(np.array(novas))
        y_lista.append(y)

    return np.concatenate(X_lista), np.concatenate(y_lista)


# ---------------- Treino ----------------

def main():
    args = parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    minds = carregar(args.minds, "minds", args.timesteps)
    nucleo = carregar(args.nucleo, "nucleo", args.timesteps)
    unido = unir(minds, nucleo)

    np.savez_compressed(args.salvar_npz, **unido)
    print(f"Dados unidos salvos em {args.salvar_npz}")

    classes, contagens = np.unique(unido["y"], return_counts=True)
    print(f"\n{len(classes)} classes, {len(unido['y'])} amostras no total:")
    for c, n in zip(classes, contagens):
        print(f"  {c}: {n}")

    if args.so_unir:
        return

    import tensorflow as tf
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras.layers import (
        LSTM, Bidirectional, Dense, Dropout, Masking,
    )
    from tensorflow.keras.models import Sequential

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    X = np.concatenate(
        [unido[k].reshape(len(unido["y"]), args.timesteps, -1) for k in CHAVES_X],
        axis=-1,
    ).astype(np.float32)

    print(f"\nFormato de entrada: {X.shape}")

    encoder = LabelEncoder()
    y = encoder.fit_transform(unido["y"])
    num_classes = len(encoder.classes_)

    idx = np.arange(len(y))
    idx_tv, idx_teste = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    idx_treino, idx_val = train_test_split(
        idx_tv, test_size=0.2, random_state=args.seed, stratify=y[idx_tv]
    )

    print(f"Treino: {len(idx_treino)} | Validacao: {len(idx_val)} | Teste: {len(idx_teste)}")

    X_treino, y_treino = aplicar_augmentation(
        X[idx_treino], y[idx_treino], args.augment_copies
    )
    print(f"Treino apos augmentation: {len(X_treino)} amostras")

    # As bases tem tamanhos diferentes: os pesos evitam que a maior domine.
    pesos = compute_class_weight(
        "balanced", classes=np.unique(y_treino), y=y_treino
    )
    class_weight = {int(c): float(w) for c, w in zip(np.unique(y_treino), pesos)}

    model = Sequential([
        Masking(mask_value=0.0, input_shape=(X.shape[1], X.shape[2])),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    model.fit(
        X_treino, y_treino,
        validation_data=(X[idx_val], y[idx_val]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        verbose=1,
    )

    _, acc = model.evaluate(X[idx_teste], y[idx_teste], verbose=0)
    pred = np.argmax(model.predict(X[idx_teste], verbose=0), axis=1)

    linhas = [f"Acuracia no teste (geral): {acc:.4f}"]

    origem_teste = unido["origem"][idx_teste]

    for nome in ("minds", "nucleo"):
        mascara = origem_teste == nome

        if mascara.any():
            a = float((pred[mascara] == y[idx_teste][mascara]).mean())
            linhas.append(f"Acuracia no teste ({nome}, {int(mascara.sum())} amostras): {a:.4f}")

    relatorio = classification_report(
        y[idx_teste], pred,
        labels=list(range(num_classes)),
        target_names=encoder.classes_,
        zero_division=0,
    )

    texto = "\n".join(linhas) + "\n\n" + relatorio
    print("\n" + texto)

    base = args.output[:-len(".keras")] if args.output.endswith(".keras") else args.output

    model.save(base + ".keras")

    with open(base + "_labels.json", "w", encoding="utf-8") as f:
        json.dump(encoder.classes_.tolist(), f, ensure_ascii=False, indent=2)

    with open(base + "_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "use_face": True,
            "face_norm": True,
            "trajectory": True,
            "velocity": False,
            "n_features": int(X.shape[2]),
            "timesteps": int(X.shape[1]),
        }, f, indent=2)

    with open(base + "_relatorio.txt", "w", encoding="utf-8") as f:
        f.write(texto)

    print(f"\nSalvos: {base}.keras, _labels.json, _config.json, _relatorio.txt")


if __name__ == "__main__":
    main()
    