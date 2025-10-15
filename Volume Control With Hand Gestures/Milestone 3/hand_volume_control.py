"""
Milestone 3:
----------------------------------------------------
Features:
✅ Detect one hand and draw skeleton.
✅ Calculate distance between thumb & index.
✅ Map distance to system volume (0–99%) using pycaw.
✅ Detect gestures:
    - Closed hand → mute (0%)
    - Fully open hand → max volume (100%)
    - Pinching → adjust volume smoothly (0–99%)
✅ Display gesture name: "Hand Opened", "Closed", "Pinching (Volume: XX%)"

Modules used:
- cv2, mediapipe, math, numpy, pycaw, comtypes

Note:
- Works on Windows (pycaw).
- Install requirements:
    pip install opencv-python mediapipe pycaw comtypes numpy
"""

import cv2
import mediapipe as mp
import time
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# -------------------------
# Configurations
# -------------------------
WEBCAM_INDEX = 0
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.6

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=3)

# -------------------------
# Initialize Volume Control
# -------------------------
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()  # typically (-65.25, 0.0)
min_vol, max_vol = vol_range[0], vol_range[1]

# Helper to get current volume in percentage
def get_volume_percent():
    vol_db = volume.GetMasterVolumeLevel()
    return np.interp(vol_db, [min_vol, max_vol], [0, 100])

# -------------------------
# Helper Functions
# -------------------------
def fingers_up(hand_landmarks):
    """Determine which fingers are up."""
    lm = hand_landmarks.landmark
    fingers = []

    # Thumb
    if lm[4].x < lm[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other four fingers
    tip_ids = [8, 12, 16, 20]
    for id in tip_ids:
        if lm[id].y < lm[id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers


def run_volume_control():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    prev_time = 0
    current_volume = get_volume_percent()
    previous_set_volume = current_volume
    smoothness = 5

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE
    ) as hands:

        print("Starting webcam. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            h, w, _ = frame.shape

            gesture_text = ""
            distance = 0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        drawing_spec, drawing_spec
                    )

                    # Thumb tip (4) and index tip (8)
                    thumb_tip = hand_landmarks.landmark[4]
                    index_tip = hand_landmarks.landmark[8]
                    x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
                    x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

                    # Draw thumb-index points & line
                    cv2.circle(frame, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                    cv2.circle(frame, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # Calculate distance between thumb and index
                    distance = math.hypot(x2 - x1, y2 - y1)
                    cv2.putText(frame, f"Distance: {int(distance)} px", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                    fingers = fingers_up(hand_landmarks)
                    total_fingers = fingers.count(1)

                    # Gesture recognition logic
                    if total_fingers == 0:
                        # Closed hand → mute
                        volume.SetMasterVolumeLevel(min_vol, None)
                        current_volume = 0
                        gesture_text = "Hand Closed (Muted)"

                    elif total_fingers == 5:
                        # Fully open hand → max volume
                        volume.SetMasterVolumeLevel(max_vol, None)
                        current_volume = 100
                        gesture_text = "Hand Opened (Max Volume)"

                    else:
                        # Pinching gesture → adjust volume
                        vol_percent = np.interp(distance, [20, 200], [0, 99])
                        vol_percent = np.clip(vol_percent, 0, 99)
                        current_volume = previous_set_volume + (vol_percent - previous_set_volume) / smoothness
                        previous_set_volume = current_volume

                        vol_db = np.interp(current_volume, [0, 100], [min_vol, max_vol])
                        volume.SetMasterVolumeLevel(vol_db, None)
                        gesture_text = f"Pinching (Volume: {int(current_volume)}%)"

            # FPS calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0.0
            prev_time = curr_time

            # Display overlays
            cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, gesture_text, (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Milestone 3 - Volume Control (Updated)', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_volume_control()
