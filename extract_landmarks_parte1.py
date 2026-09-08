"""
Extrator de landmarks de mao para o dataset de alfabeto estatico da Libras.

Espera uma estrutura de pastas assim:
    dataset/
        A/ img1.jpg img2.jpg ...
        B/ img1.jpg img2.jpg ...
        ...

Para cada imagem, roda o MediaPipe HandLandmarker (API Tasks, que substituiu
a antiga mp.solutions.hands), extrai os 21 landmarks (x, y, z) e salva tudo
em um unico arquivo .npz com dois arrays:
    X -> shape (N, 21, 3)
    y -> shape (N,)  com os nomes das classes (letras)

Requer o arquivo de modelo hand_landmarker.task na mesma pasta (ou informado
via --model_path). Para baixar:
    wget -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

Uso:
    python extract_landmarks.py --dataset_dir ./dataset --output landmarks.npz
"""

import argparse
import os
import sys
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm


def create_detector(model_path):
    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def extract_landmarks_from_mp_image(detector, mp_image):
    """Roda o HandLandmarker em um mp.Image ja carregado e retorna o vetor (21, 3) ou None."""
    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    # Usa a primeira mao detectada
    hand = result.hand_landmarks[0]
    landmarks = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32
    )
    return landmarks


def extract_landmarks_from_image(detector, image_path):
    """Roda o HandLandmarker em uma imagem de arquivo e retorna o vetor (21, 3) ou None."""
    try:
        image = mp.Image.create_from_file(image_path)
    except Exception:
        return None

    return extract_landmarks_from_mp_image(detector, image)


def normalize_landmarks(landmarks):
    """Centraliza no pulso (ponto 0) e escala pela maior distancia entre pontos.
    Isso deixa o vetor invariante a posicao e distancia da mao em relacao a camera.
    """
    wrist = landmarks[0].copy()
    centered = landmarks - wrist
    max_dist = np.linalg.norm(centered, axis=1).max()
    if max_dist > 0:
        centered = centered / max_dist
    return centered


def find_class_dirs(dataset_dir):
    """Localiza as pastas de classe (uma por letra), aceitando tanto
    dataset_dir/<letra>/ quanto dataset_dir/<split>/<letra>/ (ex.: train/, test/).
    Retorna uma lista de tuplas (letra, caminho_da_pasta).
    """
    entries = sorted(
        d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))
    )

    # Caso direto: dataset_dir/<letra>/
    has_images_directly = any(
        f.lower().endswith((".jpg", ".jpeg", ".png"))
        for entry in entries
        for f in os.listdir(os.path.join(dataset_dir, entry))
        if os.path.isfile(os.path.join(dataset_dir, entry, f))
    )
    if has_images_directly:
        return [(entry, os.path.join(dataset_dir, entry)) for entry in entries]

    # Caso com splits: dataset_dir/<split>/<letra>/
    class_dirs = []
    for split in entries:
        split_path = os.path.join(dataset_dir, split)
        for label in sorted(os.listdir(split_path)):
            label_path = os.path.join(split_path, label)
            if os.path.isdir(label_path):
                class_dirs.append((label, label_path))
    return class_dirs


def build_dataset(dataset_dir, model_path):
    X, y = [], []
    failed = 0

    print("Localizando pastas de classe...", flush=True)
    class_dirs = find_class_dirs(dataset_dir)
    print(f"{len(class_dirs)} pastas de classe encontradas (incluindo splits, se houver).", flush=True)

    print("Carregando o modelo HandLandmarker...", flush=True)
    start = time.time()
    detector = create_detector(model_path)
    print(f"Modelo carregado em {time.time() - start:.1f}s.", flush=True)

    total_files = sum(
        len([f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        for _, d in class_dirs
    )
    print(f"Total de imagens a processar: {total_files}", flush=True)

    start = time.time()
    with tqdm(total=total_files, file=sys.stdout, desc="Extraindo landmarks") as pbar:
        for label, class_dir in class_dirs:
            files = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            pbar.set_postfix(classe=label, ok=len(X), falhas=failed)

            for filename in files:
                path = os.path.join(class_dir, filename)
                landmarks = extract_landmarks_from_image(detector, path)

                if landmarks is None:
                    failed += 1
                else:
                    X.append(normalize_landmarks(landmarks))
                    y.append(label)

                pbar.update(1)
                pbar.set_postfix(classe=label, ok=len(X), falhas=failed)

    elapsed = time.time() - start
    print(f"\nProcessamento concluido em {elapsed / 60:.1f} minutos.")
    print(f"Total de amostras extraidas: {len(X)}")
    print(f"Total de falhas (mao nao detectada): {failed}")

    return np.array(X, dtype=np.float32), np.array(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, help="Pasta raiz do dataset (uma subpasta por letra, direto ou dentro de train/test)")
    parser.add_argument("--output", default="landmarks.npz", help="Arquivo .npz de saida")
    parser.add_argument("--model_path", default="hand_landmarker.task", help="Caminho do modelo HandLandmarker (.task)")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Modelo nao encontrado em {args.model_path}. Baixe com:\n"
            "wget -O hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )

    X, y = build_dataset(args.dataset_dir, args.model_path)
    np.savez_compressed(args.output, X=X, y=y)
    print(f"Salvo em {args.output}")


if __name__ == "__main__":
    main()