import base64
import json
import os
import time

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

from libras.landmarks import (
    create_detector,
    create_hand_detector,
    create_face_detector,
    create_pose_detector,
)
from extract_landmarks_imagem import extract_landmarks_from_mp_image
from libras.landmarks import normalize_landmarks
from libras.features import normalize_face, wrist_trajectory
from extract_landmarks_video_hands_face_minds_libras import (
    extract_hands_from_frame,
    extract_face_from_frame,
    extract_shoulders_from_frame,
    crop_face_region,
    RIGHT_EYE_CORNER_SLOT,
    LEFT_EYE_CORNER_SLOT,
)
from train_lstm_hands_face import build_points, to_features

# Todos os caminhos de arquivo sao resolvidos a partir da pasta deste
# arquivo, nao do diretorio de onde o comando "python app.py" e rodado -
# assim o app funciona igual rodando de qualquer lugar.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def caminho(*partes):
    return os.path.join(BASE_DIR, *partes)


app = Flask(__name__)

# ---------------- Estatico (alfabeto) ----------------

MODEL_PATH = caminho("modelo_alfabeto.keras")
HAND_MODEL_PATH = caminho("hand_landmarker.task")
LANDMARKS_PATH = caminho("landmarks.npz")

print("Carregando modelo de classificacao (estatico)...")
model = tf.keras.models.load_model(MODEL_PATH)

labels_path = os.path.splitext(MODEL_PATH)[0] + "_labels.json"
if os.path.exists(labels_path):
    with open(labels_path) as f:
        LABELS = json.load(f)
else:
    data = np.load(LANDMARKS_PATH, allow_pickle=True)
    LABELS = sorted(set(data["y"].tolist()))

print(f"Classes (estatico): {LABELS}")

print("Carregando o HandLandmarker (estatico)...")
detector = create_detector(HAND_MODEL_PATH)

# ---------------- Dinamico (palavras) ----------------

MODELO_DINAMICO_PATH = caminho("modelo_palavras_minds_v3.keras")
FACE_MODEL_PATH = caminho("face_landmarker.task")
POSE_MODEL_PATH = caminho("pose_landmarker.task")
FRAMES_POR_VIDEO = 30

print("Carregando modelo de classificacao (dinamico)...")
modelo_dinamico = tf.keras.models.load_model(MODELO_DINAMICO_PATH)

labels_dinamico_path = os.path.splitext(MODELO_DINAMICO_PATH)[0] + "_labels.json"
with open(labels_dinamico_path) as f:
    LABELS_DINAMICO = json.load(f)

print(f"Classes (dinamico): {LABELS_DINAMICO}")

# Representacao usada no treino (_config.json salvo ao lado do modelo).
# Sem o arquivo, vale o padrao do treino: rosto + trajetoria + velocidade.
config_dinamico_path = os.path.splitext(MODELO_DINAMICO_PATH)[0] + "_config.json"
CONFIG_DINAMICO = {"use_face": True, "use_traj": True, "use_velocity": True}
if os.path.exists(config_dinamico_path):
    with open(config_dinamico_path) as f:
        CONFIG_DINAMICO.update(json.load(f))
print(f"Representacao (dinamico): {CONFIG_DINAMICO}")

print("Carregando HandLandmarker, FaceLandmarker e PoseLandmarker (dinamico)...")
hand_detector_dinamico = create_hand_detector(HAND_MODEL_PATH, num_hands=2)
face_detector_dinamico = create_face_detector(FACE_MODEL_PATH, num_faces=1)
# O PoseLandmarker so e usado para os ombros (referencia da trajetoria do punho)
pose_detector_dinamico = create_pose_detector(POSE_MODEL_PATH) if CONFIG_DINAMICO["use_traj"] else None

buffer_hands, buffer_face, buffer_traj = [], [], []
last_shoulders = np.zeros((2, 2), dtype=np.float32)


def limpar_buffer():
    global buffer_hands, buffer_face, buffer_traj, last_shoulders
    buffer_hands, buffer_face, buffer_traj = [], [], []
    last_shoulders = np.zeros((2, 2), dtype=np.float32)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    image_b64 = payload["image"].split(",")[1]
    image_bytes = base64.b64decode(image_b64)

    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    landmarks = extract_landmarks_from_mp_image(detector, mp_image)

    if landmarks is None:
        return jsonify({"detected": False})

    normalized = normalize_landmarks(landmarks)
    input_batch = np.expand_dims(normalized, axis=0)  # (1, 21, 3)
    probs = model.predict(input_batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    return jsonify({
        "detected": True,
        "label": LABELS[pred_idx],
        "confidence": float(probs[pred_idx]),
    })


@app.route("/predict_dinamico", methods=["POST"])
def predict_dinamico():
    global last_shoulders

    payload = request.get_json()
    image_b64 = payload["image"].split(",")[1]
    image_bytes = base64.b64decode(image_b64)

    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_h, frame_w = frame_bgr.shape[:2]
    timestamp_ms = int(time.time() * 1000)

    hands_arr, wrists_xy = extract_hands_from_frame(hand_detector_dinamico, frame_rgb, timestamp_ms)

    face_crop = crop_face_region(frame_bgr)
    crop_h, crop_w = face_crop.shape[:2]
    face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_arr = extract_face_from_frame(face_detector_dinamico, face_crop_rgb, timestamp_ms)
    if face_arr.any():
        face_norm = normalize_face(face_arr, crop_w, crop_h, RIGHT_EYE_CORNER_SLOT, LEFT_EYE_CORNER_SLOT)
    else:
        face_norm = np.zeros_like(face_arr)  # rosto nao detectado fica zerado, como no treino

    traj = np.zeros((2, 2), dtype=np.float32)
    if pose_detector_dinamico is not None:
        shoulders = extract_shoulders_from_frame(pose_detector_dinamico, frame_rgb, timestamp_ms)
        if shoulders.any():
            last_shoulders = shoulders  # ombros quase nao mudam: reaproveita se a pose falhar
        traj = wrist_trajectory(wrists_xy, last_shoulders, frame_w, frame_h)

    buffer_hands.append(hands_arr)
    buffer_face.append(face_norm)
    buffer_traj.append(traj)

    if len(buffer_hands) < FRAMES_POR_VIDEO:
        return jsonify({"ready": False, "frames": len(buffer_hands), "total": FRAMES_POR_VIDEO})

    X_hands = np.array(buffer_hands, dtype=np.float32)[None]  # (1, 30, 2, 21, 3)
    X_face = np.array(buffer_face, dtype=np.float32)[None]    # (1, 30, P, 3)
    X_traj = np.array(buffer_traj, dtype=np.float32)[None]    # (1, 30, 2, 2)
    limpar_buffer()

    # Mesmas features do treino: pontos -> velocidade -> achatado
    points = build_points(X_hands, X_face, X_traj, CONFIG_DINAMICO["use_face"], CONFIG_DINAMICO["use_traj"])
    sequence = to_features(points, CONFIG_DINAMICO["use_velocity"])  # (1, 30, 816)

    probs = modelo_dinamico.predict(sequence, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    return jsonify({
        "ready": True,
        "label": LABELS_DINAMICO[pred_idx],
        "confidence": float(probs[pred_idx]),
    })


@app.route("/reset_dinamico", methods=["POST"])
def reset_dinamico():
    limpar_buffer()
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
    