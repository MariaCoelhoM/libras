import base64
import json
import os
import time

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

from extract_landmarks_imagem import create_detector, extract_landmarks_from_mp_image, normalize_landmarks
from extract_landmarks_video_hands_face_minds_libras import (
    create_hand_detector,
    create_face_detector,
    extract_hands_from_frame,
    extract_face_from_frame,
    crop_face_region,
)

app = Flask(__name__)

# ---------------- Estatico (alfabeto) ----------------

MODEL_PATH = "modelo_alfabeto.keras"
HAND_MODEL_PATH = "hand_landmarker.task"
LANDMARKS_PATH = "landmarks.npz"

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

MODELO_DINAMICO_PATH = "modelo_palavras_minds.keras"
FACE_MODEL_PATH = "face_landmarker.task"
FRAMES_POR_VIDEO = 30

print("Carregando modelo de classificacao (dinamico)...")
modelo_dinamico = tf.keras.models.load_model(MODELO_DINAMICO_PATH)

labels_dinamico_path = os.path.splitext(MODELO_DINAMICO_PATH)[0] + "_labels.json"
with open(labels_dinamico_path) as f:
    LABELS_DINAMICO = json.load(f)

print(f"Classes (dinamico): {LABELS_DINAMICO}")

print("Carregando HandLandmarker e FaceLandmarker (dinamico)...")
hand_detector_dinamico = create_hand_detector(HAND_MODEL_PATH, num_hands=2)
face_detector_dinamico = create_face_detector(FACE_MODEL_PATH, num_faces=1)

buffer_frames = []


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
    global buffer_frames

    payload = request.get_json()
    image_b64 = payload["image"].split(",")[1]
    image_bytes = base64.b64decode(image_b64)

    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    timestamp_ms = int(time.time() * 1000)

    hands_arr = extract_hands_from_frame(hand_detector_dinamico, frame_rgb, timestamp_ms)

    face_crop_rgb = cv2.cvtColor(crop_face_region(frame_bgr), cv2.COLOR_BGR2RGB)
    face_arr = extract_face_from_frame(face_detector_dinamico, face_crop_rgb, timestamp_ms)

    frame_features = np.concatenate([hands_arr.reshape(-1), face_arr.reshape(-1)])
    buffer_frames.append(frame_features)

    if len(buffer_frames) < FRAMES_POR_VIDEO:
        return jsonify({"ready": False, "frames": len(buffer_frames), "total": FRAMES_POR_VIDEO})

    sequence = np.expand_dims(np.array(buffer_frames, dtype=np.float32), axis=0)  # (1, 30, 402)
    buffer_frames = []

    probs = modelo_dinamico.predict(sequence, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    return jsonify({
        "ready": True,
        "label": LABELS_DINAMICO[pred_idx],
        "confidence": float(probs[pred_idx]),
    })


@app.route("/reset_dinamico", methods=["POST"])
def reset_dinamico():
    global buffer_frames
    buffer_frames = []
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)