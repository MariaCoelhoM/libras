"""Representacao dos sinais (so NumPy, sem MediaPipe).

Tudo que define COMO o landmark vira feature mora aqui, para que extracao,
treino, matriz.py e app.py usem exatamente a mesma conta. Se cada um
calculasse por conta propria, o app ao vivo poderia divergir do treino sem
dar erro (o modelo so receberia numeros de outra escala).

Tres ideias:

1. Configuracao de mao (ja existia, em libras.landmarks.normalize_landmarks):
   cada mao centrada no proprio punho e dividida pela maior distancia. Isso
   guarda so a FORMA da mao.

2. Trajetoria: posicao do punho em relacao ao corpo, em unidades de largura
   de ombros. Isso devolve o ponto de articulacao (onde a mao esta: perto do
   rosto, no peito...) e, olhando quadro a quadro, o movimento.

3. Face normalizada: relativa ao centro dos olhos e dividida pela distancia
   entre eles. Tira a posicao absoluta do rosto no recorte (que poderia
   servir de atalho para reconhecer o sinalizador) e a escala (distancia da
   camera).

Unidades: os landmarks do MediaPipe vem normalizados por largura (x) e altura
(y) da imagem, entao 1 unidade em x nao e do mesmo tamanho que 1 unidade em y.
Antes de medir distancias, as funcoes abaixo convertem para "pixels" usando a
largura/altura reais. Sem isso, o mesmo gesto daria numeros diferentes numa
webcam 4:3 e num video 16:9.
"""

import numpy as np


# ---------------- Face ----------------

def normalize_face(face_arr, width_px, height_px, right_corner_slot, left_corner_slot):
    """Normaliza a face em relacao ao proprio rosto.

    face_arr: (P, 3), coordenadas normalizadas do MediaPipe relativas ao
        recorte em que a face foi detectada (zeros se nao houve deteccao).
    width_px, height_px: tamanho do recorte em pixels.
    right_corner_slot / left_corner_slot: posicao, dentro dos P pontos, dos
        cantos externos dos olhos (mesh 33 e 263).

    Devolve (P, 3): (ponto - centro dos olhos) / distancia entre os cantos
    externos. Mantem zeros se o rosto nao foi detectado.
    """
    if not face_arr.any():
        return np.zeros_like(face_arr)

    pts = face_arr.astype(np.float32).copy()
    # z do MediaPipe tem escala parecida com a de x
    pts[:, 0] *= width_px
    pts[:, 1] *= height_px
    pts[:, 2] *= width_px

    right = pts[right_corner_slot]
    left = pts[left_corner_slot]
    center = (right + left) / 2.0
    scale = np.linalg.norm(left[:2] - right[:2])
    if scale < 1e-6:
        return np.zeros_like(face_arr)

    return ((pts - center) / scale).astype(np.float32)


# ---------------- Trajetoria ----------------

def wrist_trajectory(wrists_xy, shoulders_xy, width_px, height_px):
    """Posicao de cada punho em relacao ao corpo, em larguras de ombro.

    wrists_xy: (2, 2) com (x, y) normalizados do punho de cada mao
        (slot 0 = Left, slot 1 = Right, como em extract_hands_from_frame).
        Mao nao detectada = (0, 0) E marcada por `wrists_xy[i].any() == False`.
    shoulders_xy: (2, 2) com (x, y) normalizados dos dois ombros
        (zeros se a pose nao foi detectada).

    Devolve (2, 2): (punho - ponto medio dos ombros) / largura dos ombros.
    Fica em zeros para a mao ausente, e para as duas maos se nao ha ombros.

    So usa x e y: o z do landmark de mao e relativo ao proprio punho, entao
    o z do punho e sempre 0 e nao informa nada sobre onde ele esta.
    """
    out = np.zeros((2, 2), dtype=np.float32)
    if not shoulders_xy.any():
        return out

    scale_px = np.array([width_px, height_px], dtype=np.float32)
    s0 = shoulders_xy[0] * scale_px
    s1 = shoulders_xy[1] * scale_px
    center = (s0 + s1) / 2.0
    shoulder_width = np.linalg.norm(s0 - s1)
    if shoulder_width < 1e-6:
        return out

    for slot in range(2):
        if wrists_xy[slot].any():
            out[slot] = (wrists_xy[slot] * scale_px - center) / shoulder_width
    return out


def add_velocity(traj):
    """Velocidade entre quadros consecutivos da trajetoria.

    traj: (N, T, 2, 2) ou (N, T, 4). Devolve o mesmo formato.
    v[t] = traj[t] - traj[t-1], com v[0] = 0. Se a mao esta ausente (zeros)
    no quadro t ou no t-1, a velocidade dela e 0: senao a diferenca entre
    "ausente" e "presente" viraria um pico falso.
    """
    original_shape = traj.shape
    t = traj.reshape(traj.shape[0], traj.shape[1], 2, 2)

    velocity = np.zeros_like(t)
    present = t.any(axis=-1)  # (N, T, 2): essa mao existe nesse quadro?
    both_present = present[:, 1:] & present[:, :-1]
    velocity[:, 1:] = (t[:, 1:] - t[:, :-1]) * both_present[..., None]

    return velocity.reshape(original_shape)


# ---------------- Montagem do vetor por quadro ----------------

def default_config():
    """Representacao 'atual' (a de antes): maos + rosto cru, sem trajetoria."""
    return {"use_face": True, "face_norm": False, "trajectory": False, "velocity": False}


def needs_pose(config):
    return bool(config["trajectory"] or config["velocity"])


def assemble_features(hands, face, face_norm, traj, config):
    """Monta (N, T, F) a partir dos arrays de sequencia.

    hands:     (N, T, 2, 21, 3)
    face:      (N, T, P, 3) cru, ou None
    face_norm: (N, T, P, 3) normalizado, ou None
    traj:      (N, T, 2, 2), ou None

    A ORDEM das partes e fixa: maos | face | trajetoria | velocidade. Treino e
    app usam esta mesma funcao, entao a ordem nunca desencontra.
    """
    N, T = hands.shape[0], hands.shape[1]
    parts = [hands.reshape(N, T, -1)]

    if config["use_face"]:
        source = face_norm if config["face_norm"] else face
        if source is None:
            raise ValueError("Config pede rosto"
                             + (" normalizado" if config["face_norm"] else "")
                             + ", mas o array correspondente nao foi fornecido.")
        parts.append(source.reshape(N, T, -1))

    if config["trajectory"] or config["velocity"]:
        if traj is None:
            raise ValueError("Config pede trajetoria/velocidade, mas 'traj' nao foi fornecido.")
        if config["trajectory"]:
            parts.append(traj.reshape(N, T, -1))
        if config["velocity"]:
            parts.append(add_velocity(traj).reshape(N, T, -1))

    return np.concatenate(parts, axis=-1).astype(np.float32)


def load_sequence_features(data_path, config):
    """Le um .npz de extracao e devolve (X, y_raw, paths, config_efetiva).

    config_efetiva e a representacao que de fato foi montada (pode ser mais
    simples que a pedida se o .npz nao tem rosto, por exemplo). E ela que o
    treino deve salvar ao lado do modelo, para o app montar as mesmas features.

    Funciona com .npz antigos (chave 'X', so maos) e novos ('X_hands', 'X_face',
    'X_face_norm', 'X_traj'). Se a representacao pedida precisa de uma chave que
    o arquivo nao tem, avisa qual re-extracao fazer.
    """
    data = np.load(data_path, allow_pickle=True)
    y_raw = data["y"]
    paths = data["paths"] if "paths" in data else None

    if "X_hands" not in data:
        if "X" not in data:
            raise KeyError(f"{data_path} nao tem 'X_hands' nem 'X' - formato de .npz nao reconhecido.")
        if config["use_face"] or config["face_norm"] or config["trajectory"] or config["velocity"]:
            print("Formato antigo detectado (so maos): rosto/trajetoria indisponiveis, usando so maos.")
        X_raw = data["X"]
        N, T = X_raw.shape[0], X_raw.shape[1]
        hands_only = {"use_face": False, "face_norm": False, "trajectory": False, "velocity": False}
        return X_raw.reshape(N, T, -1).astype(np.float32), y_raw, paths, hands_only

    hands = data["X_hands"]

    face = data["X_face"] if "X_face" in data else None
    face_norm = data["X_face_norm"] if "X_face_norm" in data else None
    traj = data["X_traj"] if "X_traj" in data else None

    effective = dict(config)
    if effective["use_face"] and face is None:
        print("'X_face' nao encontrado no .npz - usando so maos.")
        effective["use_face"] = False
    if effective["use_face"] and effective["face_norm"] and face_norm is None:
        raise KeyError(f"{data_path} nao tem 'X_face_norm' (face normalizada). "
                       "Re-extraia com a versao atual de extract_landmarks_video_hands_face_minds_libras.py.")
    if needs_pose(effective) and traj is None:
        raise KeyError(f"{data_path} nao tem 'X_traj' (trajetoria). "
                       "Re-extraia com --trajectory em extract_landmarks_video_hands_face_minds_libras.py.")

    X = assemble_features(hands, face, face_norm, traj, effective)
    print(f"Representacao: {effective} -> {X.shape[-1]} features por frame")
    return X, y_raw, paths, effective


def load_static_features(data_path):
    """Alfabeto (extract_landmarks_imagem.py): X (N, 21, 3), y, e a divisao original
    train/test de cada amostra ('split'), ou None em .npz antigos."""
    data = np.load(data_path, allow_pickle=True)
    split_names = data["split"] if "split" in data else None
    return data["X"], data["y"], split_names
