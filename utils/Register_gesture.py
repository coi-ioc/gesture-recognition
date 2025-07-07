import cv2
import mediapipe as mp
import csv
import os

# Inizializza MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Nome del file CSV
GESTURE_NAME = "I"  # Cambia per ogni gesto
CSV_FOLDER = "data"
CSV_FILE = f"{CSV_FOLDER}/{GESTURE_NAME}.csv"

# Crea la directory se non esiste
os.makedirs(CSV_FOLDER, exist_ok=True)

# Funzione per salvare i landmark
def save_landmarks_to_csv(landmarks):
    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        row = [GESTURE_NAME]  # La prima colonna è la classe
        for lm in landmarks:
            row.extend([lm.x, lm.y, lm.z])
        writer.writerow(row)

# Inizializza la webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Converti in RGB e processa
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Disegna i landmark se presenti
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Salva i landmark quando premi 's'
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    save_landmarks_to_csv(hand_landmarks.landmark)
                    print(f"Landmark salvati per {GESTURE_NAME}!")

        cv2.imshow("Registrazione Gesture", frame)

        # Esci con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
