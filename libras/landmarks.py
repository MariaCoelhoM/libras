"""Funcoes compartilhadas entre todos os extratores de landmarks e o app.py.

Antes, create_detector/normalize_landmarks/sample_frame_indices e a leitura de
zip/CSV estavam copiadas (com pequenas diferencas) em varios arquivos
extract_landmarks_*.py. Isso e o que causava inconsistencias quando um era
corrigido e o outro nao. Agora moram so aqui.
"""

import contextlib
import csv
import os
import sys
import zipfile

import cv2
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision


# ---------------- Normalizacao ----------------

def normalize_landmarks(landmarks):
    """Centraliza no primeiro ponto (punho) e escala pela maior distancia."""
    wrist = landmarks[0].copy()
    centered = landmarks - wrist
    max_dist = np.linalg.norm(centered, axis=1).max()
    if max_dist > 0:
        centered = centered / max_dist
    return centered


# ---------------- Detectores MediaPipe ----------------

def create_detector(model_path, num_hands=1):
    """HandLandmarker em modo IMAGEM (um frame isolado por vez)."""
    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=num_hands,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def create_hand_detector(model_path, num_hands=2):
    """HandLandmarker em modo VIDEO (sequencia, exige timestamp crescente)."""
    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def create_face_detector(model_path, num_faces=1):
    """FaceLandmarker em modo VIDEO (sequencia, exige timestamp crescente)."""
    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=num_faces,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def create_pose_detector(model_path, num_poses=1):
    """PoseLandmarker em modo VIDEO (so usado para os ombros; exige timestamp crescente)."""
    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=num_poses,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


# ---------------- Amostragem de frames de video ----------------

def sample_frame_indices(total_frames, n_samples):
    if total_frames <= n_samples:
        return list(range(total_frames))
    return sorted(set(np.linspace(0, total_frames - 1, n_samples).astype(int).tolist()))


def count_actual_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0
    count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    return count, fps


# ---------------- Rotacao de video (metadado nem sempre aplicado pelo OpenCV) ----------------

def get_rotation(video_path):
    cap = cv2.VideoCapture(video_path)
    rotation = 0
    try:
        if hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
            rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    except Exception:
        rotation = 0
    cap.release()
    return rotation


def apply_rotation(frame, rotation):
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


# ---------------- Pastas de classe para datasets de video ----------------

def find_class_dirs_video(dataset_dir):
    """dataset_dir/<classe>/*.mp4 - sem nocao de split train/test.

    Usado pelos extratores de video (MINDS-Libras, V-Librasil em pasta de
    classes). Diferente do find_class_dirs do alfabeto (que rastreia
    train/test), por isso fica separado em vez de compartilhado.
    """
    entries = sorted(
        d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))
    )
    return [(entry, os.path.join(dataset_dir, entry)) for entry in entries]


# ---------------- Leitura de anotacoes / zip ----------------

def load_annotations(csv_path, name_column="video_name", label_column="class"):
    mapping = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row[name_column].strip()] = row[label_column].strip()
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


# ---------------- Silenciar avisos nativos (ffmpeg/mediapipe) ----------------

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
