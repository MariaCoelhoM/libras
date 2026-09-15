import argparse
import contextlib
import csv
import os
import re
import sys
import tempfile
import time
import zipfile
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

from extract_landmarks import create_detector, normalize_landmarks, find_class_dirs

FRAMES_POR_VIDEO = 30

_FACEMESH_LIPS = frozenset([
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
    (17, 314), (314, 405), (405, 321), (321, 375),
    (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
    (37, 0), (0, 267),
    (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
    (14, 317), (317, 402), (402, 318), (318, 324),
    (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
    (82, 13), (13, 312), (312, 311), (311, 310),
    (310, 415), (415, 308),
])
_FACEMESH_LEFT_EYE = frozenset([
    (263, 249), (249, 390), (390, 373), (373, 374),
    (374, 380), (380, 381), (381, 382), (382, 362),
    (263, 466), (466, 388), (388, 387), (387, 386),
    (386, 385), (385, 384), (384, 398), (398, 362),
])
_FACEMESH_LEFT_EYEBROW = frozenset([
    (276, 283), (283, 282), (282, 295),
    (295, 285), (300, 293), (293, 334),
    (334, 296), (296, 336),
])
_FACEMESH_RIGHT_EYE = frozenset([
    (33, 7), (7, 163), (163, 144), (144, 145),
    (145, 153), (153, 154), (154, 155), (155, 133),
    (33, 246), (246, 161), (161, 160), (160, 159),
    (159, 158), (158, 157), (157, 173), (173, 133),
])
_FACEMESH_RIGHT_EYEBROW = frozenset([
    (46, 53), (53, 52), (52, 65), (65, 55),
    (70, 63), (63, 105), (105, 66), (66, 107),
])


def get_enm_landmark_indices():
    """Uniao (ordenada, sem duplicata) dos indices dos grupos acima."""
    indices = set()
    for group in (_FACEMESH_LIPS, _FACEMESH_LEFT_EYE, _FACEMESH_LEFT_EYEBROW,
                  _FACEMESH_RIGHT_EYE, _FACEMESH_RIGHT_EYEBROW):
        for a, b in group:
            indices.add(a)
            indices.add(b)
    return sorted(indices)


ENM_LANDMARK_INDICES = get_enm_landmark_indices()
N_FACE_POINTS = len(ENM_LANDMARK_INDICES)


def create_face_detector(model_path, num_faces=1):
    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=num_faces,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


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
            mapping[row["video_name"].strip()] = row["class"].strip()
    return mapping


def label_from_filename(filename, filename_regex):
    match = filename_regex.match(filename)
    return match.group(1) if match else None


def resolve_label(filename, annotations=None, filename_regex=None):
    if annotations is not None:
        return annotations.get(filename)
    if filename_regex is not None:
        return label_from_filename(filename, filename_regex)
    return None


def find_video_members_in_zip(zip_path, annotations=None, filename_regex=None):
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

    use_filename_label = annotations is not None or filename_regex is not None

    entries = []
    skipped = 0
    for name in names:
        if name.endswith("/") or not name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue

        if use_filename_label:
            basename = os.path.basename(name)
            label = resolve_label(basename, annotations=annotations, filename_regex=filename_regex)
            if label is None:
                skipped += 1
                continue
        else:
            parts = name.strip("/").split("/")
            label = parts[-2] if len(parts) >= 2 else "unknown"

        entries.append((label, name))

    if use_filename_label and skipped:
        print(f"Aviso: {skipped} video(s) no zip nao identificados (sem correspondencia no CSV/regex, ignorados).", flush=True)

    return entries


def find_videos_flat_dir(dataset_dir, annotations=None, filename_regex=None):
    video_paths = []
    skipped = 0
    for filename in os.listdir(dataset_dir):
        if not filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue
        label = resolve_label(filename, annotations=annotations, filename_regex=filename_regex)
        if label is None:
            skipped += 1
            continue
        video_paths.append((label, os.path.join(dataset_dir, filename)))

    if skipped:
        print(f"Aviso: {skipped} video(s) na pasta nao identificados (sem correspondencia no CSV/regex, ignorados).", flush=True)

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


def extract_face_from_frame(detector, frame_rgb):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)

    face_arr = np.zeros((N_FACE_POINTS, 3), dtype=np.float32)

    if not result.face_landmarks:
        return face_arr

    all_landmarks = result.face_landmarks[0]  # so 1 rosto (num_faces=1)
    for out_idx, mesh_idx in enumerate(ENM_LANDMARK_INDICES):
        lm = all_landmarks[mesh_idx]
        face_arr[out_idx] = [lm.x, lm.y, lm.z]

    return face_arr


def count_actual_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    return count


def extract_sequence_from_video(hand_detector, face_detector, video_path, n_frames=FRAMES_POR_VIDEO):
    total_frames = count_actual_frames(video_path)
    if total_frames <= 0:
        return None, None, "total_frames_invalido"

    indices = sample_frame_indices(total_frames, n_frames)
    target_set = set(indices)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, "nao_abriu"

    hand_sequence = np.zeros((n_frames, 2, 21, 3), dtype=np.float32)
    face_sequence = np.zeros((n_frames, N_FACE_POINTS, 3), dtype=np.float32)

    any_hand_detected = False
    any_face_detected = False
    seq_idx = 0
    frame_idx = 0

    while seq_idx < len(indices):
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx in target_set:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hands_arr = extract_hands_from_frame(hand_detector, frame_rgb)
            if hands_arr.any():
                any_hand_detected = True
            hand_sequence[seq_idx] = hands_arr

            face_arr = extract_face_from_frame(face_detector, frame_rgb)
            if face_arr.any():
                any_face_detected = True
            face_sequence[seq_idx] = face_arr

            seq_idx += 1

        frame_idx += 1

    cap.release()

    if seq_idx < len(indices):
        return None, None, "leitura_interrompida"

    if not any_hand_detected:
        return None, None, "nenhuma_mao_detectada"

    if not any_face_detected:
        return None, None, "nenhum_rosto_detectado"

    return hand_sequence, face_sequence, None


def build_dataset_from_zip(zip_path, hand_model_path, face_model_path, annotations=None,
                            filename_regex=None, quiet=False, failures_log=None):
    X_hands, X_face, y, paths = [], [], [], []
    failed = 0

    print("Listando videos dentro do zip...", flush=True)
    entries = find_video_members_in_zip(zip_path, annotations=annotations, filename_regex=filename_regex)
    print(f"Total de videos a processar: {len(entries)}", flush=True)

    print("Carregando os modelos HandLandmarker (2 maos) e FaceLandmarker...", flush=True)
    hand_detector = create_detector(hand_model_path, num_hands=2)
    face_detector = create_face_detector(face_model_path, num_faces=1)

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
            with tqdm(total=len(entries), file=sys.stdout, desc="Extraindo maos+rosto") as pbar, noise_guard:
                for label, member in entries:
                    pbar.set_postfix(classe=label, ok=len(X_hands), falhas=failed)
                    basename = os.path.basename(member)

                    local_path = zf.extract(member, tmpdir)
                    hand_seq, face_seq, reason = extract_sequence_from_video(hand_detector, face_detector, local_path)
                    os.remove(local_path)

                    if hand_seq is None:
                        failed += 1
                        if fail_writer:
                            fail_writer.writerow([label, member, reason])
                            fail_f.flush()
                    else:
                        X_hands.append(hand_seq)
                        X_face.append(face_seq)
                        y.append(label)
                        paths.append(basename)

                    pbar.update(1)
                    pbar.set_postfix(classe=label, ok=len(X_hands), falhas=failed)
    finally:
        if fail_f:
            fail_f.close()
            print(f"Log de falhas salvo em {failures_log}", flush=True)

    elapsed = time.time() - start
    print(f"\nProcessamento concluido em {elapsed / 60:.1f} minutos.")
    print(f"Total de sequencias extraidas: {len(X_hands)}")
    print(f"Total de falhas: {failed}")

    return (np.array(X_hands, dtype=np.float32), np.array(X_face, dtype=np.float32),
            np.array(y), np.array(paths))


def build_dataset(dataset_dir, hand_model_path, face_model_path, annotations=None,
                   filename_regex=None, quiet=False, failures_log=None):
    X_hands, X_face, y, paths = [], [], [], []
    failed = 0

    if annotations is not None or filename_regex is not None:
        print("Resolvendo classes dos videos na pasta (CSV/regex)...", flush=True)
        video_paths = find_videos_flat_dir(dataset_dir, annotations=annotations, filename_regex=filename_regex)
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

    print("Carregando os modelos HandLandmarker (2 maos) e FaceLandmarker...", flush=True)
    hand_detector = create_detector(hand_model_path, num_hands=2)
    face_detector = create_face_detector(face_model_path, num_faces=1)

    noise_guard = redirect_native_stderr_to_devnull() if quiet else contextlib.nullcontext()

    fail_f = open(failures_log, "w", newline="", encoding="utf-8") if failures_log else None
    fail_writer = None
    if fail_f:
        fail_writer = csv.writer(fail_f)
        fail_writer.writerow(["classe", "arquivo", "motivo"])
        fail_f.flush()

    start = time.time()
    try:
        with tqdm(total=len(video_paths), file=sys.stdout, desc="Extraindo maos+rosto") as pbar, noise_guard:
            for label, path in video_paths:
                pbar.set_postfix(classe=label, ok=len(X_hands), falhas=failed)
                basename = os.path.basename(path)

                hand_seq, face_seq, reason = extract_sequence_from_video(hand_detector, face_detector, path)
                if hand_seq is None:
                    failed += 1
                    if fail_writer:
                        fail_writer.writerow([label, path, reason])
                        fail_f.flush()
                else:
                    X_hands.append(hand_seq)
                    X_face.append(face_seq)
                    y.append(label)
                    paths.append(basename)

                pbar.update(1)
                pbar.set_postfix(classe=label, ok=len(X_hands), falhas=failed)
    finally:
        if fail_f:
            fail_f.close()
            print(f"Log de falhas salvo em {failures_log}", flush=True)

    elapsed = time.time() - start
    print(f"\nProcessamento concluido em {elapsed / 60:.1f} minutos.")
    print(f"Total de sequencias extraidas: {len(X_hands)}")
    print(f"Total de falhas: {failed}")

    return (np.array(X_hands, dtype=np.float32), np.array(X_face, dtype=np.float32),
            np.array(y), np.array(paths))


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset_dir", help="Pasta raiz do dataset ja extraido (uma subpasta por palavra, ou pasta unica se --annotations_csv/--filename_regex for usado)")
    group.add_argument("--zip_path", help="Caminho do .zip do dataset")
    parser.add_argument("--output", default="landmarks_video.npz", help="Arquivo .npz de saida")
    parser.add_argument("--model_path", default="hand_landmarker.task", help="Caminho do modelo HandLandmarker (.task)")
    parser.add_argument("--face_model_path", default="face_landmarker.task", help="Caminho do modelo FaceLandmarker (.task)")
    parser.add_argument("--annotations_csv", default=None, help="Caminho do annotations.csv (necessario para datasets com pasta unica e CSV, ex.: V-Librasil)")
    parser.add_argument("--filename_regex", default=None, help="Regex com 1 grupo de captura para extrair a classe do nome do arquivo (necessario para datasets com pasta unica e sem CSV, ex.: MINDS-Libras). Nao use junto com --annotations_csv.")
    parser.add_argument("--quiet", action="store_true", help="Silencia avisos nativos do ffmpeg/mediapipe durante a extracao. Nao afeta a barra de progresso nem as mensagens do script.")
    parser.add_argument("--failures_log", default=None, help="Caminho de um .csv onde salvar classe/arquivo/motivo de cada video que falhou (nao_abriu, total_frames_invalido, leitura_interrompida, nenhuma_mao_detectada, nenhum_rosto_detectado). Recomendado para investigar taxas de falha altas.")
    args = parser.parse_args()

    if args.annotations_csv and args.filename_regex:
        raise SystemExit("Use --annotations_csv OU --filename_regex, nao os dois ao mesmo tempo.")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Modelo de mao nao encontrado em {args.model_path}. Baixe com:\n"
            "wget -O hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
    if not os.path.exists(args.face_model_path):
        raise FileNotFoundError(
            f"Modelo de rosto nao encontrado em {args.face_model_path}. Baixe com:\n"
            "wget -O face_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )

    annotations = None
    if args.annotations_csv:
        print(f"Carregando anotacoes de {args.annotations_csv}...", flush=True)
        annotations = load_annotations(args.annotations_csv)
        print(f"{len(annotations)} entradas carregadas do CSV.", flush=True)

    filename_regex = None
    if args.filename_regex:
        filename_regex = re.compile(args.filename_regex)

    print(f"Usando {N_FACE_POINTS} pontos faciais (sobrancelhas, olhos, boca) de 468 possiveis.", flush=True)

    if args.zip_path:
        X_hands, X_face, y, paths = build_dataset_from_zip(
            args.zip_path, args.model_path, args.face_model_path,
            annotations=annotations, filename_regex=filename_regex,
            quiet=args.quiet, failures_log=args.failures_log,
        )
    else:
        X_hands, X_face, y, paths = build_dataset(
            args.dataset_dir, args.model_path, args.face_model_path,
            annotations=annotations, filename_regex=filename_regex,
            quiet=args.quiet, failures_log=args.failures_log,
        )

    np.savez_compressed(
        args.output,
        X_hands=X_hands,
        X_face=X_face,
        y=y,
        paths=paths,
        face_landmark_indices=np.array(ENM_LANDMARK_INDICES),
    )
    print(f"Salvo em {args.output}")


if __name__ == "__main__":
    main()
