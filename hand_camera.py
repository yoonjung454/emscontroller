import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

MODEL_PATH = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# 각 손가락을 이루는 관절 번호
FINGERS = {
    "Thumb": [
        (0, 1, 2),
        (1, 2, 3),
        (2, 3, 4)
    ],
    "Index": [
        (0, 5, 6),
        (5, 6, 7),
        (6, 7, 8)
    ],
    "Middle": [
        (0, 9, 10),
        (9, 10, 11),
        (10, 11, 12)
    ],
    "Ring": [
        (0, 13, 14),
        (13, 14, 15),
        (14, 15, 16)
    ],
    "Little": [
        (0, 17, 18),
        (17, 18, 19),
        (18, 19, 20)
    ]
}

# 최근 5개 값을 평균 내서 흔들림 감소
angle_history = {
    finger: deque(maxlen=5)
    for finger in FINGERS
}


def calculate_joint_angle(point_a, point_b, point_c):
    """B 관절을 중심으로 A-B-C 사이의 3차원 각도 계산"""

    a = np.array([point_a.x, point_a.y, point_a.z])
    b = np.array([point_b.x, point_b.y, point_b.z])
    c = np.array([point_c.x, point_c.y, point_c.z])

    vector_1 = a - b
    vector_2 = c - b

    denominator = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)

    if denominator == 0:
        return 180.0

    cosine = np.dot(vector_1, vector_2) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


def calculate_finger_bend(world_landmarks, joints):
    """손가락 관절 3개의 굽힘을 합산"""

    total_bend = 0

    for a, b, c in joints:
        joint_angle = calculate_joint_angle(
            world_landmarks[a],
            world_landmarks[b],
            world_landmarks[c]
        )

        # 펴진 상태 약 180도 → 굽힘 0도
        bend = 180 - joint_angle
        total_bend += max(0, bend)

    # 관절 3개의 총 굽힘각: 0~270도 범위로 제한
    return min(total_bend, 270)


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

start_time = time.monotonic()

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = camera.read()

        if not success:
            print("카메라 영상을 가져오지 못했습니다.")
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (time.monotonic() - start_time) * 1000
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        height, width, _ = frame.shape

        if result.hand_landmarks:
            image_landmarks = result.hand_landmarks[0]
            world_landmarks = result.hand_world_landmarks[0]

            points = []

            # 관절점 표시
            for landmark in image_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                points.append((x, y))
                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # 관절 연결선 표시
            for start, end in HAND_CONNECTIONS:
                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (255, 0, 0),
                    2
                )

            y_position = 35
            output_values = []

            for finger_name, joints in FINGERS.items():
                bend_angle = calculate_finger_bend(
                    world_landmarks,
                    joints
                )

                angle_history[finger_name].append(bend_angle)

                smooth_angle = sum(
                    angle_history[finger_name]
                ) / len(angle_history[finger_name])

                bend_percent = int(
                    np.clip(smooth_angle / 270 * 100, 0, 100)
                )

                output_values.append(
                    f"{finger_name}={bend_percent}"
                )

                text = (
                    f"{finger_name}: "
                    f"{smooth_angle:.0f} deg  "
                    f"{bend_percent}%"
                )

                cv2.putText(
                    frame,
                    text,
                    (20, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )

                y_position += 30

            print(" | ".join(output_values))

        else:
            cv2.putText(
                frame,
                "Hand not detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "Hand Bend Angle",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()