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

# Referencia para o roteamento automatico: movimento indica sinal dinamico (LSTM).
# A decisao efetiva e calibrada no segmentador do navegador (SEG em index.html).
LIMIAR_MOVIMENTO = 0.25

# Intervalo entre os quadros de uma palavra (a pagina captura 1 quadro a cada 100 ms)
INTERVALO_MS = 100

# Tempos de processamento no servidor, em ms (ultimas 2000 medidas de cada tipo).
# Consulte em http://localhost:5000/latencia depois de usar a demo.
LATENCIAS = {chave: deque(maxlen=2000) for chave in ("estatico", "auto", "stream", "dinamico_frame", "dinamico_final", "dinamico_total")}


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
ultimo_punho_stream = None   # (horario do quadro em ms, xy) do punho no ultimo quadro de /stream


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


@app.route("/stream", methods=["POST"])
def stream():
    """Fluxo continuo (modos Palavras e Automatico). Recebe um lote de quadros, cada um com o horario t (ms)
    medido no navegador, e devolve por quadro: se ha mao e a velocidade do punho (largura da imagem por segundo).
    A decisao letra x sinal fica no navegador (criarSegmentador em templates/index.html).
    A letra so e classificada no ultimo quadro do lote, e so se a mao estiver parada."""
    global ultimo_punho_stream
    inicio = time.perf_counter()
    dados = request.get_json()
    frames = dados["frames"]
    letras = dados.get("letras", False)
    parado = dados.get("parado", LIMIAR_MOVIMENTO)

    saida = []
    for i, f in enumerate(frames):
        landmarks = detectar_mao(decodificar(f["img"]))
        item = {"t": f["t"], "detected": landmarks is not None, "v": 0.0}
        if landmarks is None:
            ultimo_punho_stream = None
        else:
            punho = np.asarray(landmarks, dtype=np.float32).reshape(-1, 3)[0][:2]
            if ultimo_punho_stream is not None:
                t_ant, p_ant = ultimo_punho_stream
                dt = (f["t"] - t_ant) / 1000
                if 0 < dt <= 1.0:  # horario fora de ordem ou lacuna grande: sem velocidade
                    item["v"] = float(np.linalg.norm(punho - p_ant) / dt)
            ultimo_punho_stream = (f["t"], punho)
            if letras and i == len(frames) - 1 and item["v"] < parado:
                label, confianca = classificar_letra(landmarks)
                item.update({"label": label, "confidence": confianca, "accepted": confianca >= LIMIAR_ESTATICO})
        saida.append(item)
    return jsonify({"frames": saida, "latency_ms": registrar_latencia("stream", inicio)})


@app.route("/predict_palavra", methods=["POST"])
def predict_palavra():
    """Recebe os quadros de um sinal (capturados no navegador, 1 a cada 100 ms),
    extrai maos/rosto/trajetoria de cada um e classifica com a LSTM."""
    global ultimo_ts

    inicio_total = time.perf_counter()
    imagens = request.get_json()["images"]
    if not imagens:
        return jsonify({"error": "Nenhum quadro recebido."}), 400

    # A LSTM usa 30 quadros. Amostramos ANTES de decodificar e executar
    # MediaPipe, evitando processar quadros que seriam descartados depois.
    if len(imagens) > FRAMES_POR_VIDEO:
        indices = np.linspace(0, len(imagens) - 1, FRAMES_POR_VIDEO).round().astype(int)
    else:
        # Com menos de 30 quadros, processa todos com timestamps unicos;
        # a repeticao para completar 30 acontece depois da extracao.
        indices = np.arange(len(imagens), dtype=int)
    frames = [decodificar(imagens[i]) for i in indices]

    # Os detectores de video exigem timestamps sempre crescentes.
    # Mantemos o espacamento temporal original dos quadros amostrados.
    base_ms = max(int(time.time() * 1000), ultimo_ts + 1)
    ultimo_ts = base_ms + len(imagens) * INTERVALO_MS

    hands, faces, trajs = [], [], []
    tempos_mao, tempos_rosto, tempos_pose = [], [], []
    com_mao = com_rosto = com_ombros = 0
    ombros = np.zeros((2, 2), dtype=np.float32)  # ombros quase nao mudam: reaproveita se a pose falhar

    for i, frame_bgr in enumerate(frames):
        inicio = time.perf_counter()
        timestamp_ms = base_ms + int(indices[i]) * INTERVALO_MS
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame_bgr.shape[:2]

        t_etapa = time.perf_counter()
        hands_arr, wrists_xy = extract_hands_from_frame(hand_detector_dinamico, frame_rgb, timestamp_ms)
        tempos_mao.append((time.perf_counter() - t_etapa) * 1000)

        t_etapa = time.perf_counter()
        face_crop = crop_face_region(frame_bgr)
        crop_h, crop_w = face_crop.shape[:2]
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_arr = extract_face_from_frame(face_detector_dinamico, face_crop_rgb, timestamp_ms)
        if face_arr.any():
            face_norm = normalize_face(face_arr, crop_w, crop_h, RIGHT_EYE_CORNER_SLOT, LEFT_EYE_CORNER_SLOT)
        else:
            face_norm = np.zeros_like(face_arr)  # rosto nao detectado fica zerado, como no treino
        tempos_rosto.append((time.perf_counter() - t_etapa) * 1000)

        traj = np.zeros((2, 2), dtype=np.float32)
        if pose_detector_dinamico is not None:
            t_etapa = time.perf_counter()
            shoulders = extract_shoulders_from_frame(pose_detector_dinamico, frame_rgb, timestamp_ms)
            tempos_pose.append((time.perf_counter() - t_etapa) * 1000)
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

    # Completa sequencias curtas por reamostragem; sequencias longas ja foram
    # reduzidas a 30 antes da extracao, economizando inferencias do MediaPipe.
    if len(frames) < FRAMES_POR_VIDEO:
        idx = np.linspace(0, len(frames) - 1, FRAMES_POR_VIDEO).round().astype(int)
        hands = np.asarray(hands, dtype=np.float32)[idx]
        faces = np.asarray(faces, dtype=np.float32)[idx]
        trajs = np.asarray(trajs, dtype=np.float32)[idx]

    X_hands = np.asarray(hands, dtype=np.float32)[None]  # (1, 30, 2, 21, 3)
    X_face = np.asarray(faces, dtype=np.float32)[None]   # (1, 30, P, 3)
    X_traj = np.asarray(trajs, dtype=np.float32)[None]   # (1, 30, 2, 2)

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
        "latency_ms": registrar_latencia("dinamico_total", inicio_total),
        "latency_stages_ms": {
            "maos_media": round(float(np.mean(tempos_mao)), 1) if tempos_mao else 0,
            "rosto_media": round(float(np.mean(tempos_rosto)), 1) if tempos_rosto else 0,
            "pose_media": round(float(np.mean(tempos_pose)), 1) if tempos_pose else 0,
            "classificacao": round((time.perf_counter() - inicio_clf) * 1000, 1),
        },
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