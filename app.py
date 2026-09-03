"""
Servidor local para testar o modelo com a webcam de verdade, acessada pelo
navegador (a pagina em templates/index.html usa getUserMedia para pegar o
video da sua camera e manda frames para essa API classificar).

Uso:
    python app.py

Depois abra a URL que o Codespace forwardear para a porta 5000 no seu navegador.
"""

import base64
import json
import os

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

from extract_landmarks import create_detector, extract_landmarks_from_mp_image, normalize_landmarks

app = Flask(__name__)

MODEL_PATH = "modelo_alfabeto.keras"
HAND_MODEL_PATH = "hand_landmarker.task"
LANDMARKS_PATH = "landmarks.npz"

print("Carregando modelo de classificacao...")
model = tf.keras.models.load_model(MODEL_PATH)

labels_path = os.path.splitext(MODEL_PATH)[0] + "_labels.json"
if os.path.exists(labels_path):
    with open(labels_path) as f:
        LABELS = json.load(f)
else:
    data = np.load(LANDMARKS_PATH, allow_pickle=True)
    LABELS = sorted(set(data["y"].tolist()))

print(f"Classes: {LABELS}")

print("Carregando o HandLandmarker...")
detector = create_detector(HAND_MODEL_PATH)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    image_b64 = payload["image"].split(",")[1]  # remove o prefixo "data:image/jpeg;base64,"
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
