import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import pythoncom

# ------------------------- Streamlit UI -------------------------
st.set_page_config(page_title="Volume Control with Hand Gestures", layout="centered")
st.title("Volume control with hand gestures")

# Start/Stop button
run = st.button("Start Camera")
stop = st.button("Stop Camera")

# Webcam display
frame_placeholder = st.empty()

# Initialize COM for Pycaw (required for Windows)
pythoncom.CoInitialize()
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]

# ------------------------- MediaPipe Setup -------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ------------------------- Helper Functions -------------------------
def get_finger_states(lm_list):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if lm_list[tips[0]].x > lm_list[tips[0]-1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for tip in range(1,5):
        if lm_list[tips[tip]].y < lm_list[tips[tip]-2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def draw_volume_bar(img, vol_percent):
    bar_top = 50
    bar_bottom = 400
    bar_left = 50
    bar_right = 100
    # Background
    cv2.rectangle(img, (bar_left, bar_top), (bar_right, bar_bottom), (255,255,255), 2)
    # Fill
    filled = int(np.interp(vol_percent, [0,100], [bar_bottom, bar_top]))
    cv2.rectangle(img, (bar_left, filled), (bar_right, bar_bottom), (0,255,0), cv2.FILLED)
    return img

# ------------------------- Main Loop -------------------------
vol_bar = 400
smoothness = 0.8

if run:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            st.warning("Could not access the camera.")
            break

        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        vol_percent = 0
        gesture_text = "No Hand Detected"

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                lm_list = [lm for lm in handLms.landmark]
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

                # Thumb & Index
                x1, y1 = int(lm_list[4].x * w), int(lm_list[4].y * h)
                x2, y2 = int(lm_list[8].x * w), int(lm_list[8].y * h)
                cv2.circle(img, (x1,y1), 8, (255,0,255), -1)
                cv2.circle(img, (x2,y2), 8, (255,0,255), -1)
                cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)

                # Distance → Volume %
                length = math.hypot(x2 - x1, y2 - y1)
                vol_percent = np.clip(np.interp(length, [10,250], [0,99]),0,99)

                # Finger states
                fingers = get_finger_states(lm_list)
                total_fingers = fingers.count(1)

                if total_fingers == 0:
                    volume.SetMasterVolumeLevel(min_vol, None)
                    gesture_text = "Hand Closed (Muted)"
                    target_bar = 400
                elif total_fingers == 5:
                    volume.SetMasterVolumeLevel(max_vol, None)
                    vol_percent = 100
                    gesture_text = "Hand Open (100%)"
                    target_bar = 50
                else:
                    vol = np.interp(length, [10,250], [min_vol,max_vol])
                    volume.SetMasterVolumeLevel(vol, None)
                    gesture_text = "Pinching (Adjusting)"
                    target_bar = np.interp(vol_percent, [0,100], [400,50])

                # Smooth animation
                vol_bar = int(vol_bar*smoothness + target_bar*(1-smoothness))

        # Draw volume bar
        img = draw_volume_bar(img, vol_percent)

        # Draw text
        cv2.putText(img, f"Volume: {int(vol_percent)}%", (160,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
        cv2.putText(img, f"Gesture: {gesture_text}", (160,80), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

        # Show frame in Streamlit
        frame_placeholder.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Stop condition
        if stop:
            break

    cap.release()
