import cv2
import numpy as np
from collections import deque

cap = cv2.VideoCapture('ball_tracking_example.mp4')

if not cap.isOpened():
    print("Error: Video file not found or cannot be opened.")
    exit()

pts = deque(maxlen=64)

def detect_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([29, 86, 6])
    upper_bound = np.array([64, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask

def draw_ball(frame, contours):
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            pts.appendleft((cx, cy))
    return frame

def draw_path(frame, pts):
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    contours, mask = detect_ball(frame)
    frame = draw_ball(frame, contours)
    draw_path(frame, pts)

    cv2.putText(frame, f"Contours: {len(contours)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Ball Tracking", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
