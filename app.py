
import base64
import json
import os
import time
import threading
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

try:
    from frases import montar_frase
except Exception as erro:  # spaCy/modelo ausente: o resto do app continua funcionando
    print("Frases desativadas:", erro)
    montar_frase = None

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

# Todos os caminhos de arquivo sao resolvidos a partir da pasta deste arquivo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def caminho(*partes):
    return os.path.join(BASE_DIR, *partes)


app = Flask(__name__)

# A chave do Gemini é lida de GEMINI_API_KEY no ambiente;
# não a coloque neste arquivo.
print("Arquivo do servidor:", os.path.abspath(__file__))

try:
    from frases import GEMINI_MODELO

    print("Modelo Gemini configurado:", GEMINI_MODELO)
    print(
        "Chave Gemini configurada:",
        bool(os.environ.get("GEMINI_API_KEY"))
    )

    if not os.environ.get("GEMINI_API_KEY"):
        print(
            "\n*** ATENCAO: GEMINI_API_KEY nao definida, "
            "as frases vao sair sem LLM ***\n"
        )

except Exception:
    pass


# ---------------- Configurações de classificação ----------------

LIMIAR_ESTATICO = 0.70
LIMIAR_DINAMICO = 0.50
LIMIAR_MOVIMENTO = 0.25

INTERVALO_MS = 100


# ---------------- Registrador de latências ----------------

LATENCIAS = {
    chave: deque(maxlen=2000)
    for chave in (
        "estatico",
        "auto",
        "stream",
        "frase",
        "dinamico_frame",
        "dinamico_final",
        "dinamico_total",
    )
}


def registrar_latencia(chave, inicio):
    ms = (time.perf_counter() - inicio) * 1000
    LATENCIAS[chave].append(ms)
    return round(ms, 1)


# ---------------- Cache de frases ----------------

# Evita chamar o Gemini repetidamente para as mesmas glosas.
CACHE_FRASES = {}

# Impede que duas requisições de frase sejam processadas
# simultaneamente pelo mesmo processo Flask.
LOCK_FRASES = threading.Lock()

# Tempo de armazenamento dos resultados.
TEMPO_CACHE_SUCESSO = 300  # 5 minutos
TEMPO_CACHE_FALLBACK = 20  # 20 segundos

# Limite de frases armazenadas.
LIMITE_CACHE_FRASES = 100


# ---------------- Estático (Alfabeto) ----------------

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


# ---------------- Dinâmico (Palavras em Libras) ----------------

MODELO_DINAMICO_PATH = caminho(
    "modelo_palavras_com_rosto_otimizado.keras"
)

FACE_MODEL_PATH = caminho("face_landmarker.task")
POSE_MODEL_PATH = caminho("pose_landmarker.task")

FRAMES_POR_VIDEO = 30

print("Carregando modelo de classificacao (dinamico)...")
modelo_dinamico = tf.keras.models.load_model(MODELO_DINAMICO_PATH)

labels_dinamico_path = (
    os.path.splitext(MODELO_DINAMICO_PATH)[0] + "_labels.json"
)

with open(labels_dinamico_path) as f:
    LABELS_DINAMICO = json.load(f)

print(f"Classes (dinamico): {LABELS_DINAMICO}")

config_dinamico_path = (
    os.path.splitext(MODELO_DINAMICO_PATH)[0] + "_config.json"
)

CONFIG_DINAMICO = {
    "use_face": True,
    "face_norm": True,
    "trajectory": True,
    "velocity": False,
}

if os.path.exists(config_dinamico_path):
    with open(config_dinamico_path) as f:
        CONFIG_DINAMICO.update(json.load(f))

print(f"Representacao (dinamico): {CONFIG_DINAMICO}")

print(
    "Carregando HandLandmarker, FaceLandmarker "
    "e PoseLandmarker (dinamico)..."
)

hand_detector_dinamico = create_hand_detector(
    HAND_MODEL_PATH,
    num_hands=2
)

face_detector_dinamico = create_face_detector(
    FACE_MODEL_PATH,
    num_faces=1
)

pose_detector_dinamico = (
    create_pose_detector(POSE_MODEL_PATH)
    if needs_pose(CONFIG_DINAMICO)
    else None
)

ultimo_ts = 0
ultimo_punho = None
ultimo_punho_stream = None


# ---------------- Funções auxiliares ----------------

def decodificar(imagem_b64):
    dados = base64.b64decode(imagem_b64.split(",")[1])

    return cv2.imdecode(
        np.frombuffer(dados, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )


def detectar_mao(frame_bgr):
    frame_rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    return extract_landmarks_from_mp_image(
        detector,
        mp_image
    )


def classificar_letra(landmarks):
    probs = model.predict(
        np.expand_dims(
            normalize_landmarks(landmarks),
            axis=0
        ),
        verbose=0
    )[0]

    idx = int(np.argmax(probs))

    return LABELS[idx], float(probs[idx])


# ---------------- Rotas Flask ----------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    inicio = time.perf_counter()

    landmarks = detectar_mao(
        decodificar(request.get_json()["image"])
    )

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
    global ultimo_punho

    inicio = time.perf_counter()

    landmarks = detectar_mao(
        decodificar(request.get_json()["image"])
    )

    if landmarks is None:
        ultimo_punho = None

        return jsonify({
            "detected": False,
            "rota": "cnn",
            "movimento": 0.0,
            "limiar": LIMIAR_MOVIMENTO
        })

    punho = np.asarray(
        landmarks,
        dtype=np.float32
    ).reshape(-1, 3)[0][:2]

    agora = time.time()
    movimento = 0.0

    if ultimo_punho is not None:
        tempo, anterior = ultimo_punho

        movimento = float(
            np.linalg.norm(punho - anterior)
            / max(agora - tempo, 1e-3)
        )

    ultimo_punho = (agora, punho)

    resposta = {
        "detected": True,
        "movimento": movimento,
        "limiar": LIMIAR_MOVIMENTO
    }

    if movimento > LIMIAR_MOVIMENTO:
        resposta["rota"] = "lstm"

    else:
        label, confianca = classificar_letra(landmarks)

        resposta.update({
            "rota": "cnn",
            "label": label,
            "confidence": confianca,
            "accepted": confianca >= LIMIAR_ESTATICO
        })

    resposta["latency_ms"] = registrar_latencia(
        "auto",
        inicio
    )

    return jsonify(resposta)


@app.route("/stream", methods=["POST"])
def stream():
    global ultimo_punho_stream

    inicio = time.perf_counter()

    dados = request.get_json()

    frames = dados["frames"]
    letras = dados.get("letras", False)
    parado = dados.get("parado", LIMIAR_MOVIMENTO)

    saida = []

    for i, f in enumerate(frames):

        landmarks = detectar_mao(
            decodificar(f["img"])
        )

        item = {
            "t": f["t"],
            "detected": landmarks is not None,
            "v": 0.0
        }

        if landmarks is None:
            ultimo_punho_stream = None

        else:
            punho = np.asarray(
                landmarks,
                dtype=np.float32
            ).reshape(-1, 3)[0][:2]

            if ultimo_punho_stream is not None:
                t_ant, p_ant = ultimo_punho_stream

                dt = (f["t"] - t_ant) / 1000

                if 0 < dt <= 1.0:
                    item["v"] = float(
                        np.linalg.norm(punho - p_ant) / dt
                    )

            ultimo_punho_stream = (f["t"], punho)

            if (
                letras
                and i == len(frames) - 1
                and item["v"] < parado
            ):
                label, confianca = classificar_letra(landmarks)

                item.update({
                    "label": label,
                    "confidence": confianca,
                    "accepted": confianca >= LIMIAR_ESTATICO
                })

        saida.append(item)

    return jsonify({
        "frames": saida,
        "latency_ms": registrar_latencia("stream", inicio)
    })


@app.route("/predict_palavra", methods=["POST"])
def predict_palavra():
    global ultimo_ts

    inicio_total = time.perf_counter()

    imagens = request.get_json()["images"]

    if not imagens:
        return jsonify({
            "error": "Nenhum quadro recebido."
        }), 400

    if len(imagens) > FRAMES_POR_VIDEO:
        indices = np.linspace(
            0,
            len(imagens) - 1,
            FRAMES_POR_VIDEO
        ).round().astype(int)

    else:
        indices = np.arange(
            len(imagens),
            dtype=int
        )

    frames = [
        decodificar(imagens[i])
        for i in indices
    ]

    base_ms = max(
        int(time.time() * 1000),
        ultimo_ts + 1
    )

    ultimo_ts = (
        base_ms + len(imagens) * INTERVALO_MS
    )

    hands = []
    faces = []
    trajs = []

    tempos_mao = []
    tempos_rosto = []
    tempos_pose = []

    com_mao = 0
    com_rosto = 0
    com_ombros = 0

    ombros = np.zeros(
        (2, 2),
        dtype=np.float32
    )

    for i, frame_bgr in enumerate(frames):

        inicio = time.perf_counter()

        timestamp_ms = (
            base_ms + int(indices[i]) * INTERVALO_MS
        )

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB
        )

        frame_h, frame_w = frame_bgr.shape[:2]

        # Extração das mãos
        t_etapa = time.perf_counter()

        hands_arr, wrists_xy = extract_hands_from_frame(
            hand_detector_dinamico,
            frame_rgb,
            timestamp_ms
        )

        tempos_mao.append(
            (time.perf_counter() - t_etapa) * 1000
        )

        # Extração do rosto
        t_etapa = time.perf_counter()

        face_crop = crop_face_region(frame_bgr)

        crop_h, crop_w = face_crop.shape[:2]

        face_crop_rgb = cv2.cvtColor(
            face_crop,
            cv2.COLOR_BGR2RGB
        )

        face_arr = extract_face_from_frame(
            face_detector_dinamico,
            face_crop_rgb,
            timestamp_ms
        )

        if face_arr.any():
            face_norm = normalize_face(
                face_arr,
                crop_w,
                crop_h,
                RIGHT_EYE_CORNER_SLOT,
                LEFT_EYE_CORNER_SLOT
            )

        else:
            face_norm = np.zeros_like(face_arr)

        tempos_rosto.append(
            (time.perf_counter() - t_etapa) * 1000
        )

        # Extração da pose, quando necessária
        traj = np.zeros(
            (2, 2),
            dtype=np.float32
        )

        if pose_detector_dinamico is not None:

            t_etapa = time.perf_counter()

            shoulders = extract_shoulders_from_frame(
                pose_detector_dinamico,
                frame_rgb,
                timestamp_ms
            )

            tempos_pose.append(
                (time.perf_counter() - t_etapa) * 1000
            )

            if shoulders.any():
                ombros = shoulders
                com_ombros += 1

            traj = wrist_trajectory(
                wrists_xy,
                ombros,
                frame_w,
                frame_h
            )

        hands.append(hands_arr)
        faces.append(face_norm)
        trajs.append(traj)

        com_mao += bool(hands_arr.any())
        com_rosto += bool(face_arr.any())

        registrar_latencia(
            "dinamico_frame",
            inicio
        )

    if len(frames) < FRAMES_POR_VIDEO:

        idx = np.linspace(
            0,
            len(frames) - 1,
            FRAMES_POR_VIDEO
        ).round().astype(int)

        hands = np.asarray(
            hands,
            dtype=np.float32
        )[idx]

        faces = np.asarray(
            faces,
            dtype=np.float32
        )[idx]

        trajs = np.asarray(
            trajs,
            dtype=np.float32
        )[idx]

    X_hands = np.asarray(
        hands,
        dtype=np.float32
    )[None]

    X_face = np.asarray(
        faces,
        dtype=np.float32
    )[None]

    X_traj = np.asarray(
        trajs,
        dtype=np.float32
    )[None]

    inicio_clf = time.perf_counter()

    sequence = assemble_features(
        X_hands,
        X_face,
        X_face,
        X_traj,
        CONFIG_DINAMICO
    )

    probs = modelo_dinamico.predict(
        sequence,
        verbose=0
    )[0]

    top = np.argsort(probs)[::-1][:3]

    confianca = float(probs[top[0]])

    return jsonify({
        "label": LABELS_DINAMICO[top[0]],
        "confidence": confianca,
        "accepted": confianca >= LIMIAR_DINAMICO,

        "top3": [
            {
                "label": LABELS_DINAMICO[i],
                "confidence": float(probs[i])
            }
            for i in top
        ],

        "quadros": {
            "total": len(frames),
            "maos": com_mao,
            "rosto": com_rosto,
            "ombros": com_ombros
        },

        "usa_rosto": bool(CONFIG_DINAMICO["use_face"]),

        "latency_ms": registrar_latencia(
            "dinamico_total",
            inicio_total
        ),

        "latency_stages_ms": {
            "maos_media": round(
                float(np.mean(tempos_mao)), 1
            ) if tempos_mao else 0,

            "rosto_media": round(
                float(np.mean(tempos_rosto)), 1
            ) if tempos_rosto else 0,

            "pose_media": round(
                float(np.mean(tempos_pose)), 1
            ) if tempos_pose else 0,

            "classificacao": round(
                (time.perf_counter() - inicio_clf) * 1000,
                1
            ),
        },
    })


# ---------------- Tradução das glosas para português ----------------

@app.route("/frase", methods=["POST"])
def frase():

    if montar_frase is None:
        return jsonify({
            "erro": "frases desativadas (veja o terminal)"
        }), 503

    inicio = time.perf_counter()

    dados = request.get_json(silent=True) or {}
    glosas = dados.get("glosas")

    # Verifica se as glosas foram enviadas corretamente.
    if not isinstance(glosas, list) or not glosas:

        return jsonify({
            "erro": "Nenhuma glosa válida foi recebida."
        }), 400

    # Remove espaços vazios, preservando maiúsculas e minúsculas.
    glosas_validas = [
        str(glosa).strip()
        for glosa in glosas
        if str(glosa).strip()
    ]

    if not glosas_validas:

        return jsonify({
            "erro": "As glosas estão vazias."
        }), 400

    # A chave preserva a capitalização porque o frases.py
    # pode diferenciar glosas maiúsculas de palavras comuns.
    chave = tuple(glosas_validas)

    # Impede processamento simultâneo de frases.
    with LOCK_FRASES:

        agora = time.monotonic()

        # Verifica se a frase já está armazenada.
        resultado = CACHE_FRASES.get(chave)

        if resultado is not None:

            texto, confere, usou_llm, validade = resultado

            if agora < validade:

                print("Frase recuperada do cache:", chave)

                return jsonify({
                    "frase": texto,
                    "confere": confere,
                    "llm": usou_llm,
                    "cache": True,
                    "latency_ms": registrar_latencia(
                        "frase",
                        inicio
                    )
                })

            # Remove resultado expirado.
            CACHE_FRASES.pop(chave, None)

        try:

            print("Glosas recebidas:", glosas_validas)

            # Executa a montagem da frase.
            texto, confere, usou_llm = montar_frase(
                glosas_validas
            )

            # Resultados do LLM ficam armazenados por 5 minutos.
            # Resultados de fallback ficam armazenados por 20 segundos.
            if usou_llm:
                validade = (
                    time.monotonic() + TEMPO_CACHE_SUCESSO
                )
            else:
                validade = (
                    time.monotonic() + TEMPO_CACHE_FALLBACK
                )

            CACHE_FRASES[chave] = (
                texto,
                confere,
                usou_llm,
                validade
            )

            # Limita a quantidade de resultados armazenados.
            if len(CACHE_FRASES) > LIMITE_CACHE_FRASES:

                mais_antiga = next(iter(CACHE_FRASES))

                CACHE_FRASES.pop(
                    mais_antiga,
                    None
                )

            print(
                "Frase gerada. LLM utilizado:",
                usou_llm
            )

            return jsonify({
                "frase": texto,
                "confere": confere,
                "llm": usou_llm,
                "cache": False,
                "latency_ms": registrar_latencia(
                    "frase",
                    inicio
                )
            })

        except Exception as erro:

            print("Erro ao montar frase:", erro)

            return jsonify({
                "erro": "Não foi possível montar a frase.",
                "detalhes": str(erro)
            }), 500


# ---------------- Latências ----------------

@app.route("/latencia")
def latencia():

    resumo = {}

    for chave, valores in LATENCIAS.items():

        if valores:

            resumo[chave] = {
                "n": len(valores),

                "media_ms": round(
                    float(np.mean(valores)),
                    1
                ),

                "p50_ms": round(
                    float(np.percentile(valores, 50)),
                    1
                ),

                "p95_ms": round(
                    float(np.percentile(valores, 95)),
                    1
                ),
            }

    return jsonify(resumo)


# ---------------- Inicialização ----------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
