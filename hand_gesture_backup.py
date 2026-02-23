import cv2
import mediapipe as mp
import pyautogui
import math
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

last_press = 0
cooldown = 0.1   # seconds between presses
PINCH_THRESHOLD = 100  # pixels (adjust)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            # Thumb tip = 4, Index tip = 8
            thumb = hand.landmark[4]
            index = hand.landmark[8]

            x1, y1 = int(thumb.x * w), int(thumb.y * h)
            x2, y2 = int(index.x * w), int(index.y * h)

            distance = math.hypot(x2 - x1, y2 - y1)

            cv2.circle(frame, (x1, y1), 8, (0, 255, 0), -1)
            cv2.circle(frame, (x2, y2), 8, (0, 255, 0), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Pinch detected
            if distance < PINCH_THRESHOLD:
                if time.time() - last_press > cooldown:
                    pyautogui.press("space")
                    last_press = time.time()

    cv2.imshow("Hand Control", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()