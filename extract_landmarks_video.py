"""
Extrator de landmarks de video para o dataset de palavras (sinais dinamicos) da Libras.

Espera uma estrutura de pastas assim (aceita tambem com split train/test por baixo,
igual ao extract_landmarks.py):
    dataset/
        PALAVRA1/ video1.mp4 video2.mp4 ...
        PALAVRA2/ video1.mp4 video2.mp4 ...
        ...

Para cada video, amostra FRAMES_POR_VIDEO frames uniformemente ao longo da
duracao, roda o HandLandmarker com ate 2 maos por frame, e monta uma sequencia
de tamanho fixo (T, 2, 21, 3): T frames, 2 slots de mao (esquerda/direita,
zerado se a mao nao foi detectada naquele frame), 21 pontos, 3 coordenadas.

Salva em um .npz com:
    X -> shape (N, T, 2, 21, 3)
    y -> shape (N,)  com os nomes das classes (palavras)

Requer o arquivo de modelo hand_landmarker.task (mesmo do extract_landmarks.py).

Uso:
    python extract_landmarks_video.py --dataset_dir ./dataset_palavras --output landmarks_video.npz
"""

import argparse
import os
import sys
import tempfile
import time
import zipfile
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm

from extract_landmarks import create_detector, normalize_landmarks, find_class_dirs

FRAMES_POR_VIDEO = 30


def sample_frame_indices(total_frames, n_samples):
    if total_frames <= n_samples:
        return list(range(total_frames))
    return sorted(set(np.linspace(0, total_frames - 1, n_samples).astype(int).tolist()))


def extract_hands_from_frame(detector, frame_rgb):
    """Retorna um array (2, 21, 3): slot 0 = mao esquerda, slot 1 = mao direita.
    Zerado no slot correspondente se aquela mao nao foi detectada no frame.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)

    hands_arr = np.zeros((2, 21, 3), dtype=np.float32)

    if not result.hand_landmarks:
        return hands_arr

    for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        label = handedness[0].category_name  # "Left" ou "Right"
        slot = 0 if label == "Left" else 1
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
        hands_arr[slot] = normalize_landmarks(landmarks)

    return hands_arr


def extract_sequence_from_video(detector, video_path, n_frames=FRAMES_POR_VIDEO):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    indices = sample_frame_indices(total_frames, n_frames)
    sequence = np.zeros((n_frames, 2, 21, 3), dtype=np.float32)

    any_hand_detected = False
    seq_idx = 0
    frame_idx = 0
    target_set = set(indices)

    while seq_idx < len(indices):
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx in target_set:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands_arr = extract_hands_from_frame(detector, frame_rgb)
            if hands_arr.any():
                any_hand_detected = True
            sequence[seq_idx] = hands_arr
            seq_idx += 1

        frame_idx += 1

    cap.release()

    if not any_hand_detected:
        return None

    return sequence


def find_video_members_in_zip(zip_path):
    """Le a lista de arquivos do zip e infere a classe (palavra) pelo nome
    da pasta imediatamente acima do arquivo de video, igual ao find_class_dirs
    faz com pastas reais.
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

    entries = []
    for name in names:
        if name.endswith("/") or not name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue
        parts = name.strip("/").split("/")
        label = parts[-2] if len(parts) >= 2 else "unknown"
        entries.append((label, name))
    return entries


def build_dataset_from_zip(zip_path, model_path):
    """Igual a build_dataset, mas le os videos direto de dentro do .zip,
    extraindo um por vez para um arquivo temporario e apagando em seguida.
    Evita precisar de espaco em disco para o dataset inteiro descompactado.
    """
    X, y = [], []
    failed = 0

    print("Listando videos dentro do zip...", flush=True)
    entries = find_video_members_in_zip(zip_path)
    print(f"Total de videos a processar: {len(entries)}", flush=True)

    print("Carregando o modelo HandLandmarker (2 maos)...", flush=True)
    detector = create_detector(model_path, num_hands=2)

    start = time.time()
    with zipfile.ZipFile(zip_path) as zf, tempfile.TemporaryDirectory() as tmpdir:
        with tqdm(total=len(entries), file=sys.stdout, desc="Extraindo sequencias") as pbar:
            for label, member in entries:
                pbar.set_postfix(classe=label, ok=len(X), falhas=failed)

                local_path = zf.extract(member, tmpdir)
                sequence = extract_sequence_from_video(detector, local_path)
                os.remove(local_path)  # libera espaco imediatamente

                if sequence is None:
                    failed += 1
                else:
                    X.append(sequence)
                    y.append(label)

                pbar.update(1)
                pbar.set_postfix(classe=label, ok=len(X), falhas=failed)

    elapsed = time.time() - start
    print(f"\nProcessamento concluido em {elapsed / 60:.1f} minutos.")
    print(f"Total de sequencias extraidas: {len(X)}")
    print(f"Total de falhas (nenhuma mao detectada no video): {failed}")

    return np.array(X, dtype=np.float32), np.array(y)


def build_dataset(dataset_dir, model_path):
    X, y = [], []
    failed = 0

    print("Localizando pastas de classe...", flush=True)
    class_dirs = find_class_dirs(dataset_dir)
    print(f"{len(class_dirs)} pastas de classe encontradas.", flush=True)

    print("Carregando o modelo HandLandmarker (2 maos)...", flush=True)
    detector = create_detector(model_path, num_hands=2)

    video_paths = []
    for label, class_dir in class_dirs:
        for filename in os.listdir(class_dir):
            if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_paths.append((label, os.path.join(class_dir, filename)))

    print(f"Total de videos a processar: {len(video_paths)}", flush=True)

    start = time.time()
    with tqdm(total=len(video_paths), file=sys.stdout, desc="Extraindo sequencias") as pbar:
        for label, path in video_paths:
            pbar.set_postfix(classe=label, ok=len(X), falhas=failed)

            sequence = extract_sequence_from_video(detector, path)
            if sequence is None:
                failed += 1
            else:
                X.append(sequence)
                y.append(label)

            pbar.update(1)
            pbar.set_postfix(classe=label, ok=len(X), falhas=failed)

    elapsed = time.time() - start
    print(f"\nProcessamento concluido em {elapsed / 60:.1f} minutos.")
    print(f"Total de sequencias extraidas: {len(X)}")
    print(f"Total de falhas (nenhuma mao detectada no video): {failed}")

    return np.array(X, dtype=np.float32), np.array(y)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset_dir", help="Pasta raiz do dataset ja extraido (uma subpasta por palavra)")
    group.add_argument("--zip_path", help="Caminho do .zip do dataset (recomendado: evita extrair tudo em disco)")
    parser.add_argument("--output", default="landmarks_video.npz", help="Arquivo .npz de saida")
    parser.add_argument("--model_path", default="hand_landmarker.task", help="Caminho do modelo HandLandmarker (.task)")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Modelo nao encontrado em {args.model_path}. Baixe com:\n"
            "wget -O hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )

    if args.zip_path:
        X, y = build_dataset_from_zip(args.zip_path, args.model_path)
    else:
        X, y = build_dataset(args.dataset_dir, args.model_path)

    np.savez_compressed(args.output, X=X, y=y)
    print(f"Salvo em {args.output}")


if __name__ == "__main__":
    main()
