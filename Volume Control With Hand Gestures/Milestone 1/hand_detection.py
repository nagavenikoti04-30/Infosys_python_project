"""
Milestone 1 for the hands-based volume controller project:
- Open webcam
- Detect one hand using MediaPipe Hands
- Draw skeleton points (landmarks) and connections
- Annotate landmark indices
- Show FPS

Modules used:
- cv2 (OpenCV)           : webcam and display
- mediapipe              : hand detection + landmarks
- time                   : FPS calculation
- math, numpy            : available if needed in future milestones

How it works (high-level):
1. Initialize MediaPipe Hands with `max_num_hands=1`.
2. Read frames from webcam in a loop.
3. Convert BGR->RGB (MediaPipe expects RGB).
4. Run hand detection (`hands.process`).
5. If hand detected, draw landmarks and connections, annotate each landmark index.
6. Display the annotated frame (mirrored horizontally for natural UX).
7. Press 'q' to quit.

Notes:
- If MediaPipe installation is problematic on your machine, check Python version compatibility and use the official MediaPipe install instructions:
    pip install mediapipe opencv-python
  On Windows sometimes `mediapipe` wheels are not available for older/newer Python versions. If you face issues, tell me your Python version and OS and I can help troubleshoot.
- This file is intentionally self-contained and simple so Milestone 2 can import/extend it or we can add logic in the same file.
"""

import cv2
import mediapipe as mp
import time

# -------------------------
# Configuration / settings
# -------------------------
WEBCAM_INDEX = 0           # 0 is default webcam; change if you have multiple cameras
MAX_NUM_HANDS = 1          # We only need to detect one hand in this milestone
DETECTION_CONFIDENCE = 0.7 # Minimum confidence for the detection
TRACKING_CONFIDENCE = 0.6  # Minimum confidence for the tracking
DRAW_LANDMARK_INDEX = True # Annotate landmark index numbers next to each landmark

# -------------------------
# Mediapipe initialization
# -------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Optional: custom drawing styles (colors, thickness). Keep default for now.
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=3)

# -------------------------
# Helper / main function
# -------------------------
def run_hand_skeleton():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open webcam (index {WEBCAM_INDEX}).")
        return

    # For FPS calculation
    prev_time = 0
    curr_time = 0

    # Initialize MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE
    ) as hands:

        print("Starting webcam. Press 'q' to exit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from webcam.")
                break

            # Flip horizontally for natural (mirror) view
            frame = cv2.flip(frame, 1)

            # Convert the BGR image (OpenCV) to RGB (MediaPipe)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame and find hands
            results = hands.process(rgb_frame)

            # If hands are detected, draw landmarks and connections
            if results.multi_hand_landmarks:
                # We only expect at most one hand because max_num_hands=1
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw connections + landmarks using MediaPipe's utility
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        drawing_spec,  # Landmark style
                        drawing_spec   # Connection style (reuse)
                    )

                    # Optionally annotate landmark indices (0..20)
                    if DRAW_LANDMARK_INDEX:
                        h, w, _ = frame.shape
                        for idx, lm in enumerate(hand_landmarks.landmark):
                            # lm.x and lm.y are normalized to [0,1] relative to image width/height
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            # Draw a small filled circle to make indices more visible
                            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), cv2.FILLED)
                            # Put the index number next to the landmark
                            cv2.putText(frame, str(idx), (cx + 6, cy - 6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # FPS calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0.0
            prev_time = curr_time

            # Overlay FPS and instructions
            cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

            # Show the frame
            cv2.imshow('Milestone 1 - Hand Skeleton', frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_hand_skeleton()
