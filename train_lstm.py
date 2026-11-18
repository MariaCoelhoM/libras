"""Une os .npz do MINDS e do Nucleo Comum e treina um unico modelo de palavras.

Exemplo:
    python train_lstm.py --minds landmarks_minds_libras_trimmed_interpolated.npz --nucleo landmarks_nucleo_comum.npz

A divisao treino/validacao/teste e POR SINALIZADOR (nenhuma pessoa aparece em
dois lados): sinalizador do MINDS-Libras (...Sinalizador01-5.mp4) e articulador do
V-Librasil (..._Articulador20.mp4). A divisao e feita dentro de cada base.

Saidas (na pasta atual, por padrao):
    landmarks_unificado.npz              dados unidos (com "origem" e "paths")
    modelo_libras_unificado.keras        modelo
    modelo_libras_unificado_labels.json  classes
    modelo_libras_unificado_config.json  representacao (mesmas chaves do app.py)
    modelo_libras_unificado_split.npz    indices de treino/validacao/teste (lidos por
                                         matriz.py e train_baseline.py --split)
    modelo_libras_unificado_relatorio.txt  resultados do teste, por classe e por base
    resultados/<experimento>/            metricas.json, matriz_confusao.csv/png,
                                         confusoes_principais.csv, argumentos.json,
                                         contagem_amostras.csv, relatorio.txt, split.npz
"""

import argparse
import json
import os
import unicodedata

import numpy as np

from libras.training_utils import (
    REGEX_SINALIZADOR_MINDS, REGEX_SINALIZADOR_NUCLEO,
    log_counts, signer_ids_unificado, split_por_sinalizador,
)

# Chaves exigidas nos dois .npz. A ordem das features e sempre
# mãos, rosto normalizado e trajetória, a mesma dos treinos anteriores
# e a que o assemble_features do app.py espera.
CHAVES_X = ("X_hands", "X_face_norm", "X_traj")
CHAVES = CHAVES_X + ("y", "paths")  # paths = nome do video (traz o sinalizador)


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
    p.add_argument("--epochs", type=int, default=40,
                   help="Maximo de epocas (o EarlyStopping pode parar antes)")
    p.add_argument("--patience", type=int, default=10,
                   help="EarlyStopping: epocas sem melhora do val_loss antes de parar "
                        "(0 desliga e treina todas as epocas)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--augment_copies", type=int, default=5)
    p.add_argument("--test_size", type=float, default=0.2,
                   help="Fracao de SINALIZADORES no teste (por base)")
    p.add_argument("--val_size", type=float, default=0.2,
                   help="Fracao de sinalizadores do treino usada como validacao (por base)")
    p.add_argument("--signer_regex", default=REGEX_SINALIZADOR_MINDS,
                   help="Sinalizador no nome do arquivo do MINDS-Libras")
    p.add_argument("--signer_regex_nucleo", default=REGEX_SINALIZADOR_NUCLEO,
                   help="Articulador (sinalizador) no nome do arquivo do V-Librasil")
    p.add_argument("--experiment", default="lstm_unificado",
                   help="Nome da pasta em --results_dir")
    p.add_argument("--results_dir", default="resultados")
    p.add_argument("--seed", type=int, default=42,
                   help="Semente do sorteio dos sinalizadores e do modelo")
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
    dados["paths"] = np.array([str(v) for v in dados["paths"]])

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
    unido["paths"] = np.concatenate([minds["paths"], nucleo["paths"]])
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
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import (
        LSTM, Bidirectional, Dense, Dropout, Masking,
    )
    from tensorflow.keras.models import Sequential

    from libras.experiments import ExperimentLog

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

    # ---- Divisao POR SINALIZADOR (dentro de cada base) ----
    signer_ids = signer_ids_unificado(
        unido["paths"], unido["origem"], args.signer_regex, args.signer_regex_nucleo
    )
    origem = unido["origem"]
    idx = np.arange(len(y))

    idx_tv, idx_teste = split_por_sinalizador(
        signer_ids, origem, idx, args.test_size, args.seed
    )
    idx_treino, idx_val = split_por_sinalizador(
        signer_ids, origem, idx_tv, args.val_size, args.seed
    )

    grupos_treino = set(signer_ids[idx_treino].tolist())
    grupos_val = set(signer_ids[idx_val].tolist())
    grupos_teste = set(signer_ids[idx_teste].tolist())
    assert not (grupos_treino & grupos_teste), "sinalizador no treino e no teste"
    assert not (grupos_treino & grupos_val), "sinalizador no treino e na validacao"
    assert not (grupos_val & grupos_teste), "sinalizador na validacao e no teste"

    print(f"Treino: {len(idx_treino)} | Validacao: {len(idx_val)} | Teste: {len(idx_teste)}")
    print(f"Sinalizadores no teste: {sorted(grupos_teste)}")
    print(f"Sinalizadores na validacao: {sorted(grupos_val)}")

    sem_treino = sorted(set(y.tolist()) - set(y[idx_treino].tolist()))
    sem_teste = sorted(set(y.tolist()) - set(y[idx_teste].tolist()))
    if sem_treino:
        print("ATENCAO: classes sem nenhuma amostra no treino:", list(encoder.classes_[sem_treino]))
    if sem_teste:
        print("ATENCAO: classes sem nenhuma amostra no teste:", list(encoder.classes_[sem_teste]))

    counts_by_split = {}
    log_counts("treino", unido["y"][idx_treino], counts_by_split)
    log_counts("validacao", unido["y"][idx_val], counts_by_split)
    log_counts("teste", unido["y"][idx_teste], counts_by_split)

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

    # A validacao (sinalizadores que nao estao no treino nem no teste) decide quando
    # parar e devolve os pesos da melhor epoca (menor val_loss).
    usa_early_stopping = len(idx_val) > 0 and args.patience > 0
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)
    ] if usa_early_stopping else []

    historico = model.fit(
        X_treino, y_treino,
        validation_data=(X[idx_val], y[idx_val]) if len(idx_val) else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    epocas_treinadas = len(historico.history["loss"])
    if usa_early_stopping:
        melhor = int(np.argmin(historico.history["val_loss"])) + 1
        print(f"\nEarlyStopping: {epocas_treinadas} epocas treinadas, melhor val_loss na epoca {melhor}.")
        resumo_epocas = (f"Epocas: {epocas_treinadas} de {args.epochs} "
                         f"(EarlyStopping, paciencia {args.patience}, melhor epoca {melhor})")
    else:
        resumo_epocas = f"Epocas: {epocas_treinadas} de {args.epochs} (sem EarlyStopping)"

    _, acc = model.evaluate(X[idx_teste], y[idx_teste], verbose=0)
    pred = np.argmax(model.predict(X[idx_teste], verbose=0), axis=1)

    linhas = [
        f"Acuracia no teste (geral): {acc:.4f}",
        f"Divisao por sinalizador | teste: {sorted(grupos_teste)}",
        resumo_epocas,
    ]

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

    # Indices da divisao: matriz.py e train_baseline.py --split leem este arquivo,
    # assim todos avaliam EXATAMENTE o mesmo teste.
    split = dict(
        idx_treino=idx_treino, idx_val=idx_val, idx_teste=idx_teste,
        n_amostras=len(y), signer_ids=signer_ids,
    )
    np.savez(base + "_split.npz", **split)

    # Tudo versionavel em resultados/<experimento>/
    log = ExperimentLog(args.experiment, encoder.classes_)
    log.add_run(0, args.seed, y[idx_teste], pred,
                n_treino=int(len(idx_treino)), n_validacao=int(len(idx_val)),
                acuracia_keras=float(acc), epocas_treinadas=int(epocas_treinadas))
    log.print_report()
    out_dir = log.save(
        args.results_dir,
        {**vars(args), "divisao": "por sinalizador, dentro de cada base",
         "sinalizadores_teste": sorted(grupos_teste)},
        counts_by_split,
    )
    with open(os.path.join(out_dir, "relatorio.txt"), "w", encoding="utf-8") as f:
        f.write(texto)
    np.savez(os.path.join(out_dir, "split.npz"), **split)

    print(f"\nSalvos: {base}.keras, _labels.json, _config.json, _relatorio.txt, _split.npz")
    print(f"Resultados em {out_dir}")


if __name__ == "__main__":
    main()