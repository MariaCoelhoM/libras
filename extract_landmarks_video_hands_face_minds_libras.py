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
from tqdm import tqdm

from libras.landmarks import (
    create_hand_detector,
    create_face_detector,
    create_pose_detector,
    normalize_landmarks,
    sample_frame_indices,
    count_actual_frames,
    get_rotation,
    apply_rotation,
    find_class_dirs_video,
    load_annotations,
    find_video_members_in_zip,
    find_videos_flat_dir,
    redirect_native_stderr_to_devnull,
)
from libras.features import normalize_face, wrist_trajectory

FRAMES_POR_VIDEO = 30

# Com --trim, o video e amostrado com DENSE_FACTOR vezes mais frames; depois so o
# trecho com movimento e reamostrado para FRAMES_POR_VIDEO.
DENSE_FACTOR = 2

# Parametros do recorte de frames parados (heuristica; ajuste se cortar demais/de menos).
TRIM_FRACTION = 0.15   # frame "ativo" = velocidade > TRIM_FRACTION * (percentil 95 da velocidade)
TRIM_MARGIN = 2        # frames extras mantidos antes/depois do trecho ativo
TRIM_MIN_SPAN = 8      # se o trecho ativo for menor que isso, usa o video inteiro

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
    indices = set()
    for group in (_FACEMESH_LIPS, _FACEMESH_LEFT_EYE, _FACEMESH_LEFT_EYEBROW,
                  _FACEMESH_RIGHT_EYE, _FACEMESH_RIGHT_EYEBROW):
        for a, b in group:
            indices.add(a)
            indices.add(b)
    return sorted(indices)


ENM_LANDMARK_INDICES = get_enm_landmark_indices()
N_FACE_POINTS = len(ENM_LANDMARK_INDICES)

# Posicao, dentro dos N_FACE_POINTS, dos cantos externos dos olhos (mesh 33 e 263).
# Usados por normalize_face como referencia (centro e escala do rosto).
RIGHT_EYE_CORNER_SLOT = ENM_LANDMARK_INDICES.index(33)
LEFT_EYE_CORNER_SLOT = ENM_LANDMARK_INDICES.index(263)

# No MINDS-Libras (camera fixa, sinalizador de corpo inteiro), o rosto ocupa
# uma fatia pequena do frame e o detector de rosto nao acha ele no frame
# inteiro. Recortamos a regiao topo-centro (onde o rosto sempre aparece)
# antes de rodar o FaceLandmarker. Faixas testadas e confirmadas: se algum
# video tiver enquadramento diferente, ajuste essas fracoes.
FACE_CROP_TOP, FACE_CROP_BOTTOM = 0.10, 0.45
FACE_CROP_LEFT, FACE_CROP_RIGHT = 0.30, 0.70


def crop_face_region(frame):
    h, w = frame.shape[:2]
    y1, y2 = int(h * FACE_CROP_TOP), int(h * FACE_CROP_BOTTOM)
    x1, x2 = int(w * FACE_CROP_LEFT), int(w * FACE_CROP_RIGHT)
    return frame[y1:y2, x1:x2]


# ---------------- Deteccao por frame ----------------

def extract_hands_from_frame(detector, frame_rgb, timestamp_ms):
    """Devolve (hands_arr (2,21,3) normalizado pela mao, wrists_xy (2,2) cru da imagem)."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    hands_arr = np.zeros((2, 21, 3), dtype=np.float32)
    wrists_xy = np.zeros((2, 2), dtype=np.float32)

    if not result.hand_landmarks:
        return hands_arr, wrists_xy

    for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        label = handedness[0].category_name  # "Left" ou "Right"
        slot = 0 if label == "Left" else 1
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
        hands_arr[slot] = normalize_landmarks(landmarks)
        wrists_xy[slot] = landmarks[0, :2]  # punho = landmark 0, em coordenadas da imagem

    return hands_arr, wrists_xy


def extract_face_from_frame(detector, frame_rgb, timestamp_ms):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    face_arr = np.zeros((N_FACE_POINTS, 3), dtype=np.float32)

    if not result.face_landmarks:
        return face_arr

    all_landmarks = result.face_landmarks[0]
    for out_idx, mesh_idx in enumerate(ENM_LANDMARK_INDICES):
        lm = all_landmarks[mesh_idx]
        face_arr[out_idx] = [lm.x, lm.y, lm.z]

    return face_arr


def extract_shoulders_from_frame(detector, frame_rgb, timestamp_ms):
    """Ombros (landmarks 11 e 12 da pose) em coordenadas normalizadas. Zeros se nao detectou."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    shoulders = np.zeros((2, 2), dtype=np.float32)
    if not result.pose_landmarks:
        return shoulders

    pose = result.pose_landmarks[0]
    shoulders[0] = [pose[11].x, pose[11].y]
    shoulders[1] = [pose[12].x, pose[12].y]
    return shoulders


# ---------------- Recorte dos frames parados ----------------

def find_active_span(traj_dense):
    """Acha o trecho do video em que os punhos se movem.

    traj_dense: (D, 2, 2), trajetoria do punho (zeros = mao ausente).
    Devolve (inicio, fim) inclusivos, em indices do array denso. Se nao da pra
    decidir (sem trajetoria, quase sem movimento, trecho curto demais), devolve
    o video inteiro.
    """
    D = traj_dense.shape[0]
    present = traj_dense.any(axis=-1)  # (D, 2)

    speed = np.zeros((D, 2), dtype=np.float32)
    both = present[1:] & present[:-1]
    diff = np.linalg.norm(traj_dense[1:] - traj_dense[:-1], axis=-1)
    speed[1:] = diff * both

    activity = speed.max(axis=1)
    peak = np.percentile(activity, 95)
    if peak < 1e-6:
        return 0, D - 1

    active = np.where(activity > TRIM_FRACTION * peak)[0]
    if len(active) == 0:
        return 0, D - 1

    start = max(0, int(active[0]) - TRIM_MARGIN)
    end = min(D - 1, int(active[-1]) + TRIM_MARGIN)
    if end - start + 1 < TRIM_MIN_SPAN:
        return 0, D - 1
    return start, end


# ---------------- Extracao de um video ----------------

def extract_sequence_from_video(opts, video_path, n_frames=FRAMES_POR_VIDEO):
    """Devolve (resultado_dict, None) em caso de sucesso, ou (None, motivo) em caso de falha.

    resultado_dict: hands (T,2,21,3), face (T,P,3), face_norm (T,P,3),
                    traj (T,2,2) ou None, kept (fracao do video mantida).
    """
    use_pose = opts["use_pose"]
    trim = opts["trim"]

    total_frames, fps = count_actual_frames(video_path)
    if total_frames <= 0:
        return None, "total_frames_invalido"

    rotation = get_rotation(video_path)

    n_dense = n_frames * DENSE_FACTOR if trim else n_frames
    indices = sample_frame_indices(total_frames, n_dense)
    target_set = set(indices)
    D = len(indices)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "nao_abriu"

    # Instancias dedicadas por video garantem timestamps monotonicos corretos
    hand_detector = create_hand_detector(opts["hand_model"], num_hands=2)
    face_detector = create_face_detector(opts["face_model"], num_faces=1)
    pose_detector = create_pose_detector(opts["pose_model"]) if use_pose else None

    dense_hands = np.zeros((D, 2, 21, 3), dtype=np.float32)
    dense_face = np.zeros((D, N_FACE_POINTS, 3), dtype=np.float32)
    dense_face_norm = np.zeros((D, N_FACE_POINTS, 3), dtype=np.float32)
    dense_traj = np.zeros((D, 2, 2), dtype=np.float32)

    any_hand_detected = False
    any_face_detected = False
    last_shoulders = np.zeros((2, 2), dtype=np.float32)
    seq_idx = 0
    frame_idx = 0

    try:
        while seq_idx < D:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx in target_set:
                frame = apply_rotation(frame, rotation)
                frame_h, frame_w = frame.shape[:2]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = int((frame_idx / fps) * 1000)

                hands_arr, wrists_xy = extract_hands_from_frame(hand_detector, frame_rgb, timestamp_ms)
                if hands_arr.any():
                    any_hand_detected = True
                dense_hands[seq_idx] = hands_arr

                face_crop = crop_face_region(frame)
                crop_h, crop_w = face_crop.shape[:2]
                face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                face_arr = extract_face_from_frame(face_detector, face_crop_rgb, timestamp_ms)
                if face_arr.any():
                    any_face_detected = True
                dense_face[seq_idx] = face_arr
                dense_face_norm[seq_idx] = normalize_face(
                    face_arr, crop_w, crop_h, RIGHT_EYE_CORNER_SLOT, LEFT_EYE_CORNER_SLOT
                )

                if use_pose:
                    shoulders = extract_shoulders_from_frame(pose_detector, frame_rgb, timestamp_ms)
                    if shoulders.any():
                        last_shoulders = shoulders  # ombros quase nao mudam: reaproveita se a pose falhar
                    dense_traj[seq_idx] = wrist_trajectory(wrists_xy, last_shoulders, frame_w, frame_h)

                seq_idx += 1

            frame_idx += 1
    finally:
        cap.release()
        hand_detector.close()
        face_detector.close()
        if pose_detector is not None:
            pose_detector.close()

    if seq_idx < D:
        return None, "leitura_interrompida"

    if not any_hand_detected:
        return None, "nenhuma_mao_detectada"

    if not any_face_detected:
        return None, "nenhum_rosto_detectado"

    # Escolhe quais frames densos viram a sequencia final
    if trim and use_pose:
        start, end = find_active_span(dense_traj)
    else:
        start, end = 0, D - 1
    span = np.arange(start, end + 1)
    if len(span) > n_frames:
        pick = span[np.round(np.linspace(0, len(span) - 1, n_frames)).astype(int)]
    else:
        pick = span

    hands_out = np.zeros((n_frames, 2, 21, 3), dtype=np.float32)
    face_out = np.zeros((n_frames, N_FACE_POINTS, 3), dtype=np.float32)
    face_norm_out = np.zeros((n_frames, N_FACE_POINTS, 3), dtype=np.float32)
    traj_out = np.zeros((n_frames, 2, 2), dtype=np.float32)

    hands_out[:len(pick)] = dense_hands[pick]
    face_out[:len(pick)] = dense_face[pick]
    face_norm_out[:len(pick)] = dense_face_norm[pick]
    traj_out[:len(pick)] = dense_traj[pick]

    result = {
        "hands": hands_out,
        "face": face_out,
        "face_norm": face_norm_out,
        "traj": traj_out if use_pose else None,
        "kept": len(span) / D,
    }
    return result, None


# ---------------- Loop sobre os videos (compartilhado por zip e pasta) ----------------

def process_videos(items, fetch, opts, quiet=False, failures_log=None):
    """items: lista de (classe, referencia). fetch(ref) -> (caminho_local, e_temporario)."""
    X_hands, X_face, X_face_norm, X_traj, y, paths, kept = [], [], [], [], [], [], []
    failed = 0

    noise_guard = redirect_native_stderr_to_devnull() if quiet else contextlib.nullcontext()

    fail_f = open(failures_log, "w", newline="", encoding="utf-8") if failures_log else None
    fail_writer = None
    if fail_f:
        fail_writer = csv.writer(fail_f)
        fail_writer.writerow(["classe", "arquivo", "motivo"])
        fail_f.flush()

    start = time.time()
    try:
        with tqdm(total=len(items), file=sys.stdout, desc="Extraindo maos+rosto") as pbar, noise_guard:
            for label, ref in items:
                pbar.set_postfix(classe=label, ok=len(X_hands), falhas=failed)
                basename = os.path.basename(ref)

                local_path, is_temp = fetch(ref)
                try:
                    res, reason = extract_sequence_from_video(opts, local_path)
                finally:
                    if is_temp:
                        os.remove(local_path)

                if res is None:
                    failed += 1
                    if fail_writer:
                        fail_writer.writerow([label, ref, reason])
                        fail_f.flush()
                else:
                    X_hands.append(res["hands"])
                    X_face.append(res["face"])
                    X_face_norm.append(res["face_norm"])
                    if res["traj"] is not None:
                        X_traj.append(res["traj"])
                    y.append(label)
                    paths.append(basename)
                    kept.append(res["kept"])

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
    if opts["trim"] and kept:
        print(f"Fracao media do video mantida apos o recorte: {np.mean(kept):.2f} "
              f"(min {np.min(kept):.2f}, max {np.max(kept):.2f})")

    return {
        "X_hands": np.array(X_hands, dtype=np.float32),
        "X_face": np.array(X_face, dtype=np.float32),
        "X_face_norm": np.array(X_face_norm, dtype=np.float32),
        "X_traj": np.array(X_traj, dtype=np.float32) if opts["use_pose"] else None,
        "y": np.array(y),
        "paths": np.array(paths),
    }


def build_dataset_from_zip(zip_path, opts, annotations=None, filename_regex=None,
                           quiet=False, failures_log=None):
    print("Listando videos dentro do zip...", flush=True)
    entries = find_video_members_in_zip(zip_path, annotations=annotations, filename_regex=filename_regex)
    print(f"Total de videos a processar: {len(entries)}", flush=True)

    with zipfile.ZipFile(zip_path) as zf, tempfile.TemporaryDirectory() as tmpdir:
        def fetch(member):
            return zf.extract(member, tmpdir), True

        return process_videos(entries, fetch, opts, quiet=quiet, failures_log=failures_log)


def build_dataset(dataset_dir, opts, annotations=None, filename_regex=None,
                  quiet=False, failures_log=None):
    if annotations is not None or filename_regex is not None:
        print("Resolvendo classes dos videos na pasta (CSV/regex)...", flush=True)
        video_paths = find_videos_flat_dir(dataset_dir, annotations=annotations, filename_regex=filename_regex)
    else:
        print("Localizando pastas de classe...", flush=True)
        class_dirs = find_class_dirs_video(dataset_dir)
        print(f"{len(class_dirs)} pastas de classe encontradas.", flush=True)

        video_paths = []
        for label, class_dir in class_dirs:
            for filename in os.listdir(class_dir):
                if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    video_paths.append((label, os.path.join(class_dir, filename)))

    print(f"Total de videos a processar: {len(video_paths)}", flush=True)

    def fetch(path):
        return path, False

    return process_videos(video_paths, fetch, opts, quiet=quiet, failures_log=failures_log)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset_dir", help="Pasta raiz do dataset ja extraido (uma subpasta por palavra, ou pasta unica se --annotations_csv/--filename_regex for usado)")
    group.add_argument("--zip_path", help="Caminho do .zip do dataset")
    parser.add_argument("--output", default="landmarks_video.npz", help="Arquivo .npz de saida")
    parser.add_argument("--model_path", default="hand_landmarker.task", help="Caminho do modelo HandLandmarker (.task)")
    parser.add_argument("--face_model_path", default="face_landmarker.task", help="Caminho do modelo FaceLandmarker (.task)")
    parser.add_argument("--pose_model_path", default="pose_landmarker.task", help="Caminho do modelo PoseLandmarker (.task). So usado com --trajectory/--trim.")
    parser.add_argument("--trajectory", action="store_true", help="Salva X_traj: posicao do punho em relacao aos ombros (usa o PoseLandmarker). Necessario para --trajectory/--velocity no treino.")
    parser.add_argument("--trim", action="store_true", help="Descarta o inicio/fim parados do video e reamostra os 30 frames so do trecho com movimento. Implica --trajectory (o movimento e medido pela trajetoria do punho). Mais lento: detecta 2x mais frames por video.")
    parser.add_argument("--annotations_csv", default=None, help="Caminho do annotations.csv (necessario para datasets com pasta unica e CSV, ex.: V-Librasil)")
    parser.add_argument("--filename_regex", default=None, help="Regex com 1 grupo de captura para extrair a classe do nome do arquivo (necessario para datasets com pasta unica e sem CSV, ex.: MINDS-Libras). Nao use junto com --annotations_csv.")
    parser.add_argument("--quiet", action="store_true", help="Silencia avisos nativos do ffmpeg/mediapipe durante a extracao. Nao afeta a barra de progresso nem as mensagens do script.")
    parser.add_argument("--failures_log", default=None, help="Caminho de um .csv onde salvar classe/arquivo/motivo de cada video que falhou (nao_abriu, total_frames_invalido, leitura_interrompida, nenhuma_mao_detectada, nenhum_rosto_detectado). Recomendado para investigar taxas de falha altas.")
    args = parser.parse_args()

    if args.annotations_csv and args.filename_regex:
        raise SystemExit("Use --annotations_csv OU --filename_regex, nao os dois ao mesmo tempo.")

    if args.trim and not args.trajectory:
        print("--trim precisa da trajetoria do punho: ativando --trajectory automaticamente.", flush=True)
        args.trajectory = True

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
    if args.trajectory and not os.path.exists(args.pose_model_path):
        raise FileNotFoundError(
            f"Modelo de pose nao encontrado em {args.pose_model_path}. Baixe com:\n"
            "curl -o pose_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
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
    print(f"Trajetoria do punho: {'sim' if args.trajectory else 'nao'} | "
          f"Recorte de frames parados: {'sim' if args.trim else 'nao'}", flush=True)

    opts = {
        "hand_model": args.model_path,
        "face_model": args.face_model_path,
        "pose_model": args.pose_model_path,
        "use_pose": args.trajectory,
        "trim": args.trim,
    }

    if args.zip_path:
        data = build_dataset_from_zip(
            args.zip_path, opts,
            annotations=annotations, filename_regex=filename_regex,
            quiet=args.quiet, failures_log=args.failures_log,
        )
    else:
        data = build_dataset(
            args.dataset_dir, opts,
            annotations=annotations, filename_regex=filename_regex,
            quiet=args.quiet, failures_log=args.failures_log,
        )

    to_save = {
        "X_hands": data["X_hands"],
        "X_face": data["X_face"],
        "X_face_norm": data["X_face_norm"],
        "y": data["y"],
        "paths": data["paths"],
        "face_landmark_indices": np.array(ENM_LANDMARK_INDICES),
        "trimmed": np.array(args.trim),
    }
    if data["X_traj"] is not None:
        to_save["X_traj"] = data["X_traj"]

    np.savez_compressed(args.output, **to_save)
    print(f"Salvo em {args.output}")


if __name__ == "__main__":
    main()
    