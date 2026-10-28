import base64
import json
import os
import time
from collections import deque

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
from libras.features import normalize_face, wrist_trajectory, assemble_features, needs_pose
from extract_landmarks_video_hands_face_minds_libras import (
    extract_hands_from_frame,
    extract_face_from_frame,
    extract_shoulders_from_frame,
    crop_face_region,
    RIGHT_EYE_CORNER_SLOT,
    LEFT_EYE_CORNER_SLOT,
)

# Todos os caminhos de arquivo sao resolvidos a partir da pasta deste
# arquivo, nao do diretorio de onde o comando "python app.py" e rodado -
# assim o app funciona igual rodando de qualquer lugar.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def caminho(*partes):
    return os.path.join(BASE_DIR, *partes)


app = Flask(__name__)

# Abaixo do limiar a previsao nao e aceita (a demo mostra "nao reconheci").
LIMIAR_ESTATICO = 0.70
LIMIAR_DINAMICO = 0.50

# Roteamento automatico: se o punho se move mais que isso (largura da imagem por
# segundo), o app entende que e uma palavra (LSTM); parado, e uma letra (CNN).
# A demo mostra o valor medido na tela, para voce calibrar antes de apresentar.
LIMIAR_MOVIMENTO = 0.25

# Intervalo entre os quadros de uma palavra (a pagina captura 1 quadro a cada 100 ms)
INTERVALO_MS = 100

# Tempos de processamento no servidor, em ms (ultimas 2000 medidas de cada tipo).
# Consulte em http://localhost:5000/latencia depois de usar a demo.
LATENCIAS = {chave: deque(maxlen=2000) for chave in ("estatico", "auto", "dinamico_frame", "dinamico_final")}


def registrar_latencia(chave, inicio):
    ms = (time.perf_counter() - inicio) * 1000
    LATENCIAS[chave].append(ms)
    return round(ms, 1)

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

MODELO_DINAMICO_PATH = caminho("modelo_palavras_com_rosto_otimizado.keras")
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
# As features sao montadas por libras.features.assemble_features, a mesma
# funcao do treino, entao a ordem e o tamanho do vetor sempre batem.
config_dinamico_path = os.path.splitext(MODELO_DINAMICO_PATH)[0] + "_config.json"
CONFIG_DINAMICO = {"use_face": True, "face_norm": True, "trajectory": True, "velocity": False}
if os.path.exists(config_dinamico_path):
    with open(config_dinamico_path) as f:
        CONFIG_DINAMICO.update(json.load(f))
print(f"Representacao (dinamico): {CONFIG_DINAMICO}")

print("Carregando HandLandmarker, FaceLandmarker e PoseLandmarker (dinamico)...")
hand_detector_dinamico = create_hand_detector(HAND_MODEL_PATH, num_hands=2)
face_detector_dinamico = create_face_detector(FACE_MODEL_PATH, num_faces=1)
# O PoseLandmarker so e usado para os ombros (referencia da trajetoria do punho)
pose_detector_dinamico = create_pose_detector(POSE_MODEL_PATH) if needs_pose(CONFIG_DINAMICO) else None

ultimo_ts = 0                # ultimo timestamp enviado aos detectores de video (tem que crescer sempre)
ultimo_punho = None          # (tempo, x, y) do punho na ultima chamada de /predict_auto


def decodificar(imagem_b64):
    dados = base64.b64decode(imagem_b64.split(",")[1])
    return cv2.imdecode(np.frombuffer(dados, dtype=np.uint8), cv2.IMREAD_COLOR)


def detectar_mao(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    return extract_landmarks_from_mp_image(detector, mp_image)


def classificar_letra(landmarks):
    probs = model.predict(np.expand_dims(normalize_landmarks(landmarks), axis=0), verbose=0)[0]  # (1, 21, 3)
    idx = int(np.argmax(probs))
    return LABELS[idx], float(probs[idx])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    inicio = time.perf_counter()
    landmarks = detectar_mao(decodificar(request.get_json()["image"]))

    if landmarks is None:
        return jsonify({"detected": False})

    label, confianca = classificar_letra(landmarks)
    return jsonify({
        "detected": True,
        "label": label,
        "confidence": confianca,
        "accepted": confianca >= LIMIAR_ESTATICO,
        "latency_ms": registrar_latencia("estatico", inicio),
    })


@app.route("/predict_auto", methods=["POST"])
def predict_auto():
    """Roteador: mede o movimento do punho entre uma chamada e outra.
    Parado -> classifica a letra com a CNN. Em movimento -> avisa que e palavra (LSTM)."""
    global ultimo_punho
    inicio = time.perf_counter()
    landmarks = detectar_mao(decodificar(request.get_json()["image"]))

    if landmarks is None:
        ultimo_punho = None
        return jsonify({"detected": False, "rota": "cnn", "movimento": 0.0, "limiar": LIMIAR_MOVIMENTO})

    punho = np.asarray(landmarks, dtype=np.float32).reshape(-1, 3)[0][:2]
    agora = time.time()
    movimento = 0.0
    if ultimo_punho is not None:
        tempo, anterior = ultimo_punho
        movimento = float(np.linalg.norm(punho - anterior) / max(agora - tempo, 1e-3))
    ultimo_punho = (agora, punho)

    resposta = {"detected": True, "movimento": movimento, "limiar": LIMIAR_MOVIMENTO}
    if movimento > LIMIAR_MOVIMENTO:
        resposta["rota"] = "lstm"
    else:
        label, confianca = classificar_letra(landmarks)
        resposta.update({"rota": "cnn", "label": label, "confidence": confianca,
                         "accepted": confianca >= LIMIAR_ESTATICO})
    resposta["latency_ms"] = registrar_latencia("auto", inicio)
    return jsonify(resposta)


@app.route("/predict_palavra", methods=["POST"])
def predict_palavra():
    """Recebe os quadros de um sinal (capturados no navegador, 1 a cada 100 ms),
    extrai maos/rosto/trajetoria de cada um e classifica com a LSTM."""
    global ultimo_ts

    frames = [decodificar(img) for img in request.get_json()["images"]]
    # Os detectores de video exigem timestamps sempre crescentes, mesmo entre chamadas
    base_ms = max(int(time.time() * 1000), ultimo_ts + 1)
    ultimo_ts = base_ms + len(frames) * INTERVALO_MS

    hands, faces, trajs = [], [], []
    com_mao = com_rosto = com_ombros = 0
    ombros = np.zeros((2, 2), dtype=np.float32)  # ombros quase nao mudam: reaproveita se a pose falhar

    for i, frame_bgr in enumerate(frames):
        inicio = time.perf_counter()
        timestamp_ms = base_ms + i * INTERVALO_MS
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame_bgr.shape[:2]

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
                ombros = shoulders
                com_ombros += 1
            traj = wrist_trajectory(wrists_xy, ombros, frame_w, frame_h)

        hands.append(hands_arr)
        faces.append(face_norm)
        trajs.append(traj)
        com_mao += bool(hands_arr.any())
        com_rosto += bool(face_arr.any())
        registrar_latencia("dinamico_frame", inicio)

    # A LSTM espera exatamente 30 quadros: reamostra se vieram menos (ou mais)
    idx = np.linspace(0, len(frames) - 1, FRAMES_POR_VIDEO).round().astype(int)
    X_hands = np.array(hands, dtype=np.float32)[idx][None]  # (1, 30, 2, 21, 3)
    X_face = np.array(faces, dtype=np.float32)[idx][None]   # (1, 30, P, 3)
    X_traj = np.array(trajs, dtype=np.float32)[idx][None]   # (1, 30, 2, 2)

    inicio_clf = time.perf_counter()
    # Mesmas features do treino: maos | rosto normalizado | trajetoria | velocidade
    sequence = assemble_features(X_hands, X_face, X_face, X_traj, CONFIG_DINAMICO)  # (1, 30, n_features)
    probs = modelo_dinamico.predict(sequence, verbose=0)[0]
    top = np.argsort(probs)[::-1][:3]
    confianca = float(probs[top[0]])

    return jsonify({
        "label": LABELS_DINAMICO[top[0]],
        "confidence": confianca,
        "accepted": confianca >= LIMIAR_DINAMICO,
        "top3": [{"label": LABELS_DINAMICO[i], "confidence": float(probs[i])} for i in top],
        # Diagnostico: em quantos quadros cada parte foi detectada
        "quadros": {"total": len(frames), "maos": com_mao, "rosto": com_rosto, "ombros": com_ombros},
        "usa_rosto": bool(CONFIG_DINAMICO["use_face"]),
        "latency_ms": registrar_latencia("dinamico_final", inicio_clf),
    })


@app.route("/latencia")
def latencia():
    resumo = {}
    for chave, valores in LATENCIAS.items():
        if valores:
            resumo[chave] = {
                "n": len(valores),
                "media_ms": round(float(np.mean(valores)), 1),
                "p50_ms": round(float(np.percentile(valores, 50)), 1),
                "p95_ms": round(float(np.percentile(valores, 95)), 1),
            }
    return jsonify(resumo)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)