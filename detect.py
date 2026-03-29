from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import pygame
import time
import dlib
import cv2

# --- EAR CALCULATION FORMULA ---
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --- Settings ---
EAR_THRESHOLD = 0.25   
FRAME_LIMIT = 20       

# --- Alarm setup ---
pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")  
# --- dlib setup ---
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# --- LANDMARK INDEX OF EYES ---
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# --- START WEBCAM ---
cap = cv2.VideoCapture(0)
counter = 0

print("✅ Drowsiness Detector chal raha hai... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 0)

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye  = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR  = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # DRAW OUTLINE FOR EYES
        cv2.drawContours(frame, [cv2.convexHull(leftEye)],  -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

        if ear < EAR_THRESHOLD:
            counter += 1
            if counter >= FRAME_LIMIT:
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play(-1)
                cv2.putText(frame, "⚠ DROWSINESS ALERT!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            counter = 0
            pygame.mixer.music.stop()

        cv2.putText(frame, f"EAR: {ear:.2f}", (480, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Drowsiness Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()


