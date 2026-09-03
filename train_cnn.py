"""
Treina uma CNN para classificar letras do alfabeto manual da Libras
a partir dos landmarks extraidos pelo MediaPipe Hands (script extract_landmarks.py).

Cada amostra e um vetor (21, 3): 21 pontos articulares da mao, cada um com
coordenadas (x, y, z). A CNN trata isso como uma "imagem" 1D de 21 posicoes
com 3 canais, aplicando convolucoes ao longo dos pontos articulares para
aprender combinacoes locais de configuracao da mao.

Uso:
    python train_cnn.py --data landmarks.npz --output modelo_alfabeto.keras
"""

import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


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
    parser.add_argument("--data", required=True, help="Arquivo .npz gerado por extract_landmarks.py")
    parser.add_argument("--output", default="modelo_alfabeto.keras", help="Caminho para salvar o modelo treinado")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    X, y_raw = data["X"], data["y"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)

    # Divisao estratificada: 70% treino, 15% validacao, 15% teste
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print(f"Treino: {len(X_train)} | Validacao: {len(X_val)} | Teste: {len(X_test)}")
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

    # Avaliacao final no conjunto de teste
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nAcuracia no teste: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nMatriz de confusao:")
    print(confusion_matrix(y_test, y_pred))
    print("\nRelatorio de classificacao:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    model.save(args.output)
    print(f"Modelo salvo em {args.output}")


if __name__ == "__main__":
    main()
