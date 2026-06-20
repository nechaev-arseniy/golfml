"""
Golf Swing Pose Analyzer — Phase 1 Prototype
==============================================
Uses MediaPipe's Pose Landmarker (Tasks API) to extract body landmarks
from a golf swing video, calculates key swing angles, and overlays them
on an output video.

SETUP (run once):
    pip install mediapipe opencv-python numpy

    # Download the pose landmarker model (pick one):
    curl -o pose_landmarker.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
    # or "full"/"heavy" instead of "lite" for more accuracy at the cost of speed:
    # https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models

USAGE:
    python golf_pose_analyzer.py --input swing.mp4 --model pose_landmarker.task --output annotated_swing.mp4
"""

import argparse
import json

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# MediaPipe Pose landmark indices (33-point model)
# Reference: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models
LM = {
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
    "LEFT_HIP": 23, "RIGHT_HIP": 24,
    "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
}

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]


def calculate_angle(a, b, c):
    """Angle (degrees) at point b, formed by points a-b-c. Each point is (x, y)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def xy(landmarks, name, width, height):
    lm = landmarks[LM[name]]
    return [lm.x * width, lm.y * height]


def analyze_frame(landmarks, width, height):
    """Compute key golf-relevant angles for a single frame."""
    l_shoulder = xy(landmarks, "LEFT_SHOULDER", width, height)
    r_shoulder = xy(landmarks, "RIGHT_SHOULDER", width, height)
    l_hip = xy(landmarks, "LEFT_HIP", width, height)
    r_hip = xy(landmarks, "RIGHT_HIP", width, height)
    r_elbow = xy(landmarks, "RIGHT_ELBOW", width, height)
    r_wrist = xy(landmarks, "RIGHT_WRIST", width, height)
    l_knee = xy(landmarks, "LEFT_KNEE", width, height)
    l_ankle = xy(landmarks, "LEFT_ANKLE", width, height)

    shoulder_angle = np.degrees(np.arctan2(
        r_shoulder[1] - l_shoulder[1], r_shoulder[0] - l_shoulder[0]
    ))
    hip_angle = np.degrees(np.arctan2(
        r_hip[1] - l_hip[1], r_hip[0] - l_hip[0]
    ))

    hip_mid = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]
    shoulder_mid = [(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2]
    vertical_ref = [hip_mid[0], hip_mid[1] - 100]
    spine_tilt = calculate_angle(vertical_ref, hip_mid, shoulder_mid)

    elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
    knee_angle = calculate_angle(l_hip, l_knee, l_ankle)

    return {
        "shoulder_line_angle": round(float(shoulder_angle), 1),
        "hip_line_angle": round(float(hip_angle), 1),
        "spine_tilt": round(float(spine_tilt), 1),
        "right_elbow_angle": round(float(elbow_angle), 1),
        "left_knee_angle": round(float(knee_angle), 1),
    }


def draw_skeleton(frame, landmarks, width, height):
    points = {}
    for name, idx in LM.items():
        lm = landmarks[idx]
        pt = (int(lm.x * width), int(lm.y * height))
        points[idx] = pt
        cv2.circle(frame, pt, 4, (0, 255, 0), -1)

    for a_idx, b_idx in POSE_CONNECTIONS:
        if a_idx in points and b_idx in points:
            cv2.line(frame, points[a_idx], points[b_idx], (0, 200, 255), 2)


def process_video(input_path, model_path, output_path, json_path=None):
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_duration_ms = 1000.0 / fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    all_frame_metrics = []

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        timestamp_ms = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = landmarker.detect_for_video(mp_image, int(timestamp_ms))

            metrics = {"frame": frame_idx}

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                draw_skeleton(frame, landmarks, width, height)
                metrics.update(analyze_frame(landmarks, width, height))

                y0 = 30
                for i, (k, v) in enumerate(metrics.items()):
                    if k == "frame":
                        continue
                    text = f"{k}: {v} deg"
                    cv2.putText(frame, text, (10, y0 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            all_frame_metrics.append(metrics)
            out.write(frame)
            frame_idx += 1
            timestamp_ms += frame_duration_ms

    cap.release()
    out.release()

    if json_path:
        with open(json_path, "w") as f:
            json.dump(all_frame_metrics, f, indent=2)

    print(f"Processed {frame_idx} frames.")
    print(f"Annotated video saved to: {output_path}")
    if json_path:
        print(f"Per-frame metrics saved to: {json_path}")

    return all_frame_metrics


def find_key_swing_positions(metrics_list):
    """
    Naive heuristic: frame with max shoulder rotation ~ top of backswing.
    Refine once you have labeled data with ground-truth swing phases.
    """
    valid = [m for m in metrics_list if "shoulder_line_angle" in m]
    if not valid:
        return {}
    top_of_backswing = max(valid, key=lambda m: abs(m["shoulder_line_angle"]))
    return {
        "top_of_backswing_frame": top_of_backswing["frame"],
        "top_of_backswing_metrics": top_of_backswing,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Golf Swing Pose Analyzer")
    parser.add_argument("--input", required=True, help="Path to input swing video")
    parser.add_argument("--model", default="pose_landmarker.task", help="Path to MediaPipe pose_landmarker .task model")
    parser.add_argument("--output", default="annotated_swing.mp4", help="Path to save annotated video")
    parser.add_argument("--json", default="swing_metrics.json", help="Path to save per-frame metrics JSON")
    args = parser.parse_args()

    metrics = process_video(args.input, args.model, args.output, args.json)
    key_positions = find_key_swing_positions(metrics)
    print("\nKey swing position estimate (naive heuristic):")
    print(json.dumps(key_positions, indent=2))