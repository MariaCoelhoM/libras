import argparse
import json

import cv2
import numpy as np
import tensorflow as tf

from extract_landmarks_video_hands_face_minds_libras import (
    create_hand_detector,
    create_face_detector,
    extract_hands_from_frame,
    extract_face_from_frame,
    crop_face_region,
    get_rotation,
    apply_rotation,
    count_actual_frames,
    sample_frame_indices,
    FRAMES_POR_VIDEO,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="02AlunoSinalizador02-5.mp4")
    parser.add_argument("--model", default="modelo_palavras_minds.keras")
    parser.add_argument("--hand_model", default="hand_landmarker.task")
    parser.add_argument("--face_model", default="face_landmarker.task")
    args = parser.parse_args()

    labels_path = args.model.replace(".keras", "_labels.json")
    with open(labels_path) as f:
        labels = json.load(f)

    total_frames, fps = count_actual_frames(args.video)
    rotation = get_rotation(args.video)
    indices = sample_frame_indices(total_frames, FRAMES_POR_VIDEO)
    target_set = set(indices)

    hand_detector = create_hand_detector(args.hand_model, num_hands=2)
    face_detector = create_face_detector(args.face_model, num_faces=1)

    cap = cv2.VideoCapture(args.video)
    buffer_frames = []
    frame_idx = 0

    while len(buffer_frames) < len(indices):
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx in target_set:
            frame = apply_rotation(frame, rotation)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int((frame_idx / fps) * 1000)

            hands_arr = extract_hands_from_frame(hand_detector, frame_rgb, timestamp_ms)
            face_crop_rgb = cv2.cvtColor(crop_face_region(frame), cv2.COLOR_BGR2RGB)
            face_arr = extract_face_from_frame(face_detector, face_crop_rgb, timestamp_ms)

            buffer_frames.append(np.concatenate([hands_arr.reshape(-1), face_arr.reshape(-1)]))

        frame_idx += 1

    cap.release()
    hand_detector.close()
    face_detector.close()

    print(f"Frames capturados: {len(buffer_frames)} / {FRAMES_POR_VIDEO}")

    sequence = np.expand_dims(np.array(buffer_frames, dtype=np.float32), axis=0)
    model = tf.keras.models.load_model(args.model)
    probs = model.predict(sequence, verbose=0)[0]

    top3_idx = np.argsort(probs)[::-1][:3]
    print("\nTop 3 previsoes:")
    for idx in top3_idx:
        print(f"  {labels[idx]}: {probs[idx]:.3f}")


if __name__ == "__main__":
    main()