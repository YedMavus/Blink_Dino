import cv2
import mediapipe as mp
import math
import time
import random
import numpy as np

# ---------------- HAND TRACKING ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

last_press = 0
cooldown = 0.1
PINCH_THRESHOLD = 100


# ---------------- FLAPPY BIRD SETTINGS ----------------
bird_y = 300
velocity = 0

gravity = 4
jump_strength = -25

pipes = []
pipe_gap = 200
pipe_width = 80
pipe_speed = 14

score = 0
try:
    with open("highscore.txt", "r") as f:
        high_score = int(f.read())
except FileNotFoundError:
    high_score = 0

game_started = False


def spawn_pipe(screen_h, screen_w):
    top = random.randint(50, screen_h - pipe_gap - 50)
    pipes.append([screen_w, top, False])


def reset_game():
    global bird_y, velocity, pipes, score
    bird_y = 300
    velocity = 0
    pipes.clear()
    score = 0


# ---------------- DRAW CARTOON BIRD ----------------
def draw_bird(img, x, y):
    # Body
    cv2.circle(img, (x, y), 22, (0, 255, 255), -1)

    # Wing
    cv2.ellipse(img, (x - 5, y + 5), (12, 8),
                0, 0, 360, (0, 200, 200), -1)

    # Eye
    cv2.circle(img, (x + 8, y - 6), 5, (255, 255, 255), -1)
    cv2.circle(img, (x + 10, y - 6), 2, (0, 0, 0), -1)

    # Beak
    pts = np.array([[x + 22, y],
                    [x + 32, y - 4],
                    [x + 32, y + 4]], np.int32)
    cv2.fillPoly(img, [pts], (0, 165, 255))


# ---------------- WINDOW ----------------
cv2.namedWindow("Game", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Game", cv2.WND_PROP_FULLSCREEN,
                      cv2.WINDOW_FULLSCREEN)


while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cam_h, cam_w, _ = frame.shape

    # -------- HAND PROCESSING --------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    pinch = False

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            thumb = hand.landmark[4]
            index = hand.landmark[8]

            x1, y1 = int(thumb.x * cam_w), int(thumb.y * cam_h)
            x2, y2 = int(index.x * cam_w), int(index.y * cam_h)

            distance = math.hypot(x2 - x1, y2 - y1)

            # Draw tracking on webcam feed
            cv2.circle(frame, (x1, y1), 8, (0, 255, 0), -1)
            cv2.circle(frame, (x2, y2), 8, (0, 255, 0), -1)
            cv2.line(frame, (x1, y1),
                     (x2, y2), (255, 0, 0), 2)

            if distance < PINCH_THRESHOLD:
                if time.time() - last_press > cooldown:
                    pinch = True
                    last_press = time.time()

    # -------- SCREEN SETUP --------
    screen_w = 1280
    screen_h = 720

    cam_view = cv2.resize(frame,
                          (int(screen_w * 0.3), screen_h))

    game_area = np.ones(
        (screen_h, int(screen_w * 0.7), 3), dtype=np.uint8) * 255

    # =========================================================
    # WAIT FOR FIRST PINCH
    # =========================================================
    if not game_started:

        if pinch:
            game_started = True
            velocity = jump_strength

        cv2.putText(game_area, "PINCH TO START",
                    (200, 350),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 0),
                    4)

    else:
        # =====================================================
        # GAME RUNNING
        # =====================================================

        if pinch:
            velocity = jump_strength
            bird_y += velocity

        velocity += gravity
        bird_y += velocity

        # Spawn pipes
        if len(pipes) == 0 or pipes[-1][0] < int(screen_w * 0.7) - 300:
            spawn_pipe(screen_h, int(screen_w * 0.7))

        # Move pipes
        for p in pipes:
            p[0] -= pipe_speed

        pipes[:] = [p for p in pipes if p[0] > -pipe_width]

        # -------- DRAW BIRD --------
        bird_x = 120
        draw_bird(game_area, bird_x, int(bird_y))

        # -------- DRAW PIPES --------
        for p in pipes:
            x, top, scored = p

            cv2.rectangle(game_area, (x, 0),
                          (x + pipe_width, top), (0, 200, 0), -1)

            cv2.rectangle(game_area, (x, top + pipe_gap),
                          (x + pipe_width, screen_h), (0, 200, 0), -1)

            # Collision
            if bird_x + 22 > x and bird_x - 22 < x + pipe_width:
                if bird_y - 22 < top or bird_y + 22 > top + pipe_gap:
                    reset_game()
                    game_started = False

            # Score
            if x + pipe_width < bird_x and not scored:
                score += 1
                p[2] = True
                if score > high_score:
                    high_score = score

        # Out of bounds
        if bird_y > screen_h or bird_y < 0:
            reset_game()
            game_started = False

        # -------- SCORE DISPLAY --------
        cv2.putText(game_area, f"Score: {score}",
                    (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 0),
                    3)

        cv2.putText(game_area, f"High Score: {high_score}",
                    (40, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 0),
                    3)

    # -------- COMBINE --------
    final = np.hstack((cam_view, game_area))

    cv2.imshow("Game", final)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        with open("highscore.txt", "w") as f:
            f.write(str(high_score))
        break


cap.release()
cv2.destroyAllWindows()
# import cv2
# import mediapipe as mp
# import math
# import time
# import random
# import numpy as np

# # ---------------- HAND TRACKING ----------------
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=1,
#                        min_detection_confidence=0.7,
#                        min_tracking_confidence=0.7)

# cap = cv2.VideoCapture(0)

# last_press = 0
# cooldown = 0.1
# PINCH_THRESHOLD = 100


# # ---------------- FLAPPY BIRD SETTINGS ----------------
# bird_y = 300
# velocity = 0

# gravity = 4
# jump_strength = -25

# pipes = []
# pipe_gap = 200
# pipe_width = 80
# pipe_speed = 14

# score = 0
# high_score = 0

# game_started = False


# def spawn_pipe(screen_h, screen_w):
#     top = random.randint(50, screen_h - pipe_gap - 50)
#     pipes.append([screen_w, top, False])  # [x, top, scored]


# def reset_game():
#     global bird_y, velocity, pipes, score
#     bird_y = 300
#     velocity = 0
#     pipes.clear()
#     score = 0


# # ---------------- WINDOW ----------------
# cv2.namedWindow("Game", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("Game", cv2.WND_PROP_FULLSCREEN,
#                       cv2.WINDOW_FULLSCREEN)


# while True:
#     ret, frame = cap.read()
#     frame = cv2.flip(frame, 1)
#     cam_h, cam_w, _ = frame.shape

#     # -------- HAND PROCESSING --------
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(rgb)

#     pinch = False

#     if result.multi_hand_landmarks:
#         for hand in result.multi_hand_landmarks:
#             thumb = hand.landmark[4]
#             index = hand.landmark[8]

#             x1, y1 = int(thumb.x * cam_w), int(thumb.y * cam_h)
#             x2, y2 = int(index.x * cam_w), int(index.y * cam_h)

#             distance = math.hypot(x2 - x1, y2 - y1)

#             cv2.circle(frame, (x1, y1), 8, (0, 255, 0), -1)
#             cv2.circle(frame, (x2, y2), 8, (0, 255, 0), -1)
#             cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

#             if distance < PINCH_THRESHOLD:
#                 if time.time() - last_press > cooldown:
#                     pinch = True
#                     last_press = time.time()

#     # -------- SCREEN SETUP --------
#     screen_w = 1280
#     screen_h = 720

#     cam_view = cv2.resize(frame, (int(screen_w * 0.3), screen_h))
#     game_area = np.ones(
#         (screen_h, int(screen_w * 0.7), 3), dtype=np.uint8) * 255

#     # =========================================================
#     # 🟡 WAIT FOR FIRST PINCH TO START
#     # =========================================================
#     if not game_started:

#         if pinch:
#             game_started = True
#             velocity = jump_strength

#         cv2.putText(game_area, "PINCH TO START",
#                     (200, 350),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     2,
#                     (0, 0, 0),
#                     4)

#     else:
#         # =====================================================
#         # 🟢 GAME RUNNING
#         # =====================================================

#         if pinch:
#             velocity = jump_strength
#             bird_y += velocity  # instant jump feel

#         velocity += gravity
#         bird_y += velocity

#         # Spawn pipes
#         if len(pipes) == 0 or pipes[-1][0] < int(screen_w * 0.7) - 300:
#             spawn_pipe(screen_h, int(screen_w * 0.7))

#         # Move pipes
#         for p in pipes:
#             p[0] -= pipe_speed

#         pipes[:] = [p for p in pipes if p[0] > -pipe_width]

#         # -------- DRAW BIRD --------
#         bird_x = 120
#         cv2.circle(game_area, (bird_x, int(bird_y)),
#                    20, (0, 255, 255), -1)

#         # -------- DRAW PIPES --------
#         for p in pipes:
#             x, top, scored = p

#             cv2.rectangle(game_area, (x, 0),
#                           (x + pipe_width, top), (0, 200, 0), -1)

#             cv2.rectangle(game_area, (x, top + pipe_gap),
#                           (x + pipe_width, screen_h), (0, 200, 0), -1)

#             # Collision
#             if bird_x + 20 > x and bird_x - 20 < x + pipe_width:
#                 if bird_y - 20 < top or bird_y + 20 > top + pipe_gap:
#                     reset_game()
#                     game_started = False

#             # Score
#             if x + pipe_width < bird_x and not scored:
#                 score += 1
#                 p[2] = True
#                 if score > high_score:
#                     high_score = score

#         # Out of bounds
#         if bird_y > screen_h or bird_y < 0:
#             reset_game()
#             game_started = False

#         # -------- SCORE DISPLAY --------
#         cv2.putText(game_area, f"Score: {score}",
#                     (40, 80),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1.5,
#                     (0, 0, 0),
#                     3)

#         cv2.putText(game_area, f"High Score: {high_score}",
#                     (40, 140),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1.5,
#                     (0, 0, 0),
#                     3)

#     # -------- COMBINE LEFT + RIGHT --------
#     final = np.hstack((cam_view, game_area))

#     cv2.imshow("Game", final)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break


# cap.release()
# cv2.destroyAllWindows()
