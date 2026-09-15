import argparse
import contextlib
import csv
import os
import sys
import tempfile
import time
import zipfile
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm

from extract_landmarks_imagem import create_detector, normalize_landmarks, find_class_dirs

FRAMES_POR_VIDEO = 30


@contextlib.contextmanager
def redirect_native_stderr_to_devnull():
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stderr.flush()
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_fd, stderr_fd)
        os.close(devnull_fd)
        os.close(saved_fd)


def load_annotations(csv_path):
    mapping = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_name = row["video_name"].strip()
            label = row["class"].strip()
            mapping[video_name] = label
    return mapping


def find_video_members_in_zip(zip_path, annotations=None):
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

    entries = []
    skipped = 0
    for name in names:
        if name.endswith("/") or not name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue

        if annotations is not None:
            basename = os.path.basename(name)
            label = annotations.get(basename)
            if label is None:
                skipped += 1
                continue
        else:
            parts = name.strip("/").split("/")
            label = parts[-2] if len(parts) >= 2 else "unknown"

        entries.append((label, name))

    if annotations is not None and skipped:
        print(f"Aviso: {skipped} video(s) no zip nao encontrados no annotations.csv (ignorados).", flush=True)

    return entries


def find_videos_flat_dir(dataset_dir, annotations):
    video_paths = []
    skipped = 0
    for filename in os.listdir(dataset_dir):
        if not filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue
        label = annotations.get(filename)
        if label is None:
            skipped += 1
            continue
        video_paths.append((label, os.path.join(dataset_dir, filename)))

    if skipped:
        print(f"Aviso: {skipped} video(s) na pasta nao encontrados no annotations.csv (ignorados).", flush=True)

    return video_paths


def sample_frame_indices(total_frames, n_samples):
    if total_frames <= n_samples:
        return list(range(total_frames))
    return sorted(set(np.linspace(0, total_frames - 1, n_samples).astype(int).tolist()))


def extract_hands_from_frame(detector, frame_rgb):
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
        return None, "nao_abriu"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None, "total_frames_invalido"

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

    if seq_idx < len(indices):
        return None, "leitura_interrompida"

    if not any_hand_detected:
        return None, "nenhuma_mao_detectada"

    return sequence, None


def build_dataset_from_zip(zip_path, model_path, annotations=None, quiet=False, failures_log=None):
    X, y = [], []
    failed = 0

    print("Listando videos dentro do zip...", flush=True)
    entries = find_video_members_in_zip(zip_path, annotations=annotations)
    print(f"Total de videos a processar: {len(entries)}", flush=True)

    print("Carregando o modelo HandLandmarker (2 maos)...", flush=True)
    detector = create_detector(model_path, num_hands=2)

    noise_guard = redirect_native_stderr_to_devnull() if quiet else contextlib.nullcontext()

    fail_f = open(failures_log, "w", newline="", encoding="utf-8") if failures_log else None
    fail_writer = None
    if fail_f:
        fail_writer = csv.writer(fail_f)
        fail_writer.writerow(["classe", "arquivo", "motivo"])
        fail_f.flush()

    start = time.time()
    try:
        with zipfile.ZipFile(zip_path) as zf, tempfile.TemporaryDirectory() as tmpdir:
            with tqdm(total=len(entries), file=sys.stdout, desc="Extraindo sequencias") as pbar, noise_guard:
                for label, member in entries:
                    pbar.set_postfix(classe=label, ok=len(X), falhas=failed)

                    local_path = zf.extract(member, tmpdir)
                    sequence, reason = extract_sequence_from_video(detector, local_path)
                    os.remove(local_path)  # libera espaco imediatamente

                    if sequence is None:
                        failed += 1
                        if fail_writer:
                            fail_writer.writerow([label, member, reason])
                            fail_f.flush()
                    else:
                        X.append(sequence)
                        y.append(label)

                    pbar.update(1)
                    pbar.set_postfix(classe=label, ok=len(X), falhas=failed)
    finally:
        if fail_f:
            fail_f.close()
            print(f"Log de falhas salvo em {failures_log}", flush=True)

    elapsed = time.time() - start
    print(f"\nProcessamento concluido em {elapsed / 60:.1f} minutos.")
    print(f"Total de sequencias extraidas: {len(X)}")
    print(f"Total de falhas (nenhuma mao detectada no video): {failed}")

    return np.array(X, dtype=np.float32), np.array(y)

def build_dataset(dataset_dir, model_path, annotations=None, quiet=False, failures_log=None):
    X, y = [], []
    failed = 0

    if annotations is not None:
        print("Casando videos da pasta com o annotations.csv...", flush=True)
        video_paths = find_videos_flat_dir(dataset_dir, annotations)
    else:
        print("Localizando pastas de classe...", flush=True)
        class_dirs = find_class_dirs(dataset_dir)
        print(f"{len(class_dirs)} pastas de classe encontradas.", flush=True)

        video_paths = []
        for label, class_dir in class_dirs:
            for filename in os.listdir(class_dir):
                if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    video_paths.append((label, os.path.join(class_dir, filename)))

    print(f"Total de videos a processar: {len(video_paths)}", flush=True)

    print("Carregando o modelo HandLandmarker (2 maos)...", flush=True)
    detector = create_detector(model_path, num_hands=2)

    noise_guard = redirect_native_stderr_to_devnull() if quiet else contextlib.nullcontext()

    fail_f = open(failures_log, "w", newline="", encoding="utf-8") if failures_log else None
    fail_writer = None
    if fail_f:
        fail_writer = csv.writer(fail_f)
        fail_writer.writerow(["classe", "arquivo", "motivo"])
        fail_f.flush()

    start = time.time()
    try:
        with tqdm(total=len(video_paths), file=sys.stdout, desc="Extraindo sequencias") as pbar, noise_guard:
            for label, path in video_paths:
                pbar.set_postfix(classe=label, ok=len(X), falhas=failed)

                sequence, reason = extract_sequence_from_video(detector, path)
                if sequence is None:
                    failed += 1
                    if fail_writer:
                        fail_writer.writerow([label, path, reason])
                        fail_f.flush()
                else:
                    X.append(sequence)
                    y.append(label)

                pbar.update(1)
                pbar.set_postfix(classe=label, ok=len(X), falhas=failed)
    finally:
        if fail_f:
            fail_f.close()
            print(f"Log de falhas salvo em {failures_log}", flush=True)

    elapsed = time.time() - start
    print(f"\nProcessamento concluido em {elapsed / 60:.1f} minutos.")
    print(f"Total de sequencias extraidas: {len(X)}")
    print(f"Total de falhas (nenhuma mao detectada no video): {failed}")

    return np.array(X, dtype=np.float32), np.array(y)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset_dir", help="Pasta raiz do dataset ja extraido (uma subpasta por palavra, ou pasta unica se --annotations_csv for usado)")
    group.add_argument("--zip_path", help="Caminho do .zip do dataset (recomendado: evita extrair tudo em disco)")
    parser.add_argument("--output", default="landmarks_video.npz", help="Arquivo .npz de saida")
    parser.add_argument("--model_path", default="hand_landmarker.task", help="Caminho do modelo HandLandmarker (.task)")
    parser.add_argument("--annotations_csv", default=None, help="Caminho do annotations.csv (necessario para datasets com pasta unica, ex.: V-Librasil)")
    parser.add_argument("--quiet", action="store_true", help="Silencia avisos nativos do ffmpeg/mediapipe (ex.: 'swscaler ... interlaced') durante a extracao. Nao afeta a barra de progresso nem as mensagens do script.")
    parser.add_argument("--failures_log", default=None, help="Caminho de um .csv onde salvar classe/arquivo/motivo de cada video que falhou (nao_abriu, total_frames_invalido, leitura_interrompida, nenhuma_mao_detectada). Recomendado para investigar taxas de falha altas.")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Modelo nao encontrado em {args.model_path}. Baixe com:\n"
            "wget -O hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )

    annotations = None
    if args.annotations_csv:
        print(f"Carregando anotacoes de {args.annotations_csv}...", flush=True)
        annotations = load_annotations(args.annotations_csv)
        print(f"{len(annotations)} entradas carregadas do CSV.", flush=True)

    if args.zip_path:
        X, y = build_dataset_from_zip(args.zip_path, args.model_path, annotations=annotations, quiet=args.quiet, failures_log=args.failures_log)
    else:
        X, y = build_dataset(args.dataset_dir, args.model_path, annotations=annotations, quiet=args.quiet, failures_log=args.failures_log)

    np.savez_compressed(args.output, X=X, y=y)
    print(f"Salvo em {args.output}")


if __name__ == "__main__":
    main()
    