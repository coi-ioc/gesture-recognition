import cv2
import mediapipe as mp
import torch
from inference.predict import predict
from models.model import GestureRecognitionModel
import pickle
import numpy as np
from collections import deque

# Percorsi
MODEL_PATH = "gesture_recognition_model.pth"
SCALER_ENCODER_PATH = "scaler_encoder.pkl"

# Carica scaler ed encoder dal file unico
with open(SCALER_ENCODER_PATH, "rb") as f:
    scaler, encoder = pickle.load(f)

# Inizializza MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Carica il modello
input_size = 63  # 21 landmark x 3 coordinate
num_classes = len(encoder.classes_)
model = GestureRecognitionModel(input_size, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Variabili per il controllo dei gesti
current_gesture = None
gesture_queue = deque(maxlen=15)  # Coda per i gesti rilevati
phrase = ""  # Frase costruita

# Inizializza la webcam
cap = cv2.VideoCapture(0)
# Inizializza MediaPipe Hands una volta per tutto, invece di ogni frame
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            # Per ogni mano rilevata (prendiamo ad esempio la prima)
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Estrai i landmark dall'ultimo hand_landmarks processato
            landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
            input_data = np.array([landmarks])
            
            # Predizione: la funzione predict restituisce una stringa
            predicted_label = str(predict(input_data, MODEL_PATH, SCALER_ENCODER_PATH))
            
            # Aggiungi il gesto alla coda
            gesture_queue.append(predicted_label)
            
            # Trova il gesto predominante nella coda
            most_common_gesture = max(set(gesture_queue), key=gesture_queue.count)
            current_gesture = most_common_gesture

        # Visualizza il gesto corrente sul frame
        cv2.putText(frame, f"Gesto: {current_gesture}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Gesture Recognition", frame)

        # Visualizza la frase in una finestra separata
        phrase_frame = np.ones((200, 800, 3), dtype=np.uint8) * 255  # Sfondo bianco
        cv2.putText(phrase_frame, f"Frase: {phrase}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Frase", phrase_frame)

        # Gestione dei tasti:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Esci con 'q'
            break
        elif key == 32:  # Spazio: conferma e aggiungi il gesto corrente alla frase
            if current_gesture is not None and current_gesture != "delete":
                phrase += current_gesture
        elif key == 8:  # Backspace: cancella l'ultimo carattere
            phrase = phrase[:-1]
        elif key == 27:  # Esc: cancella l'intera frase
            phrase = ""

cap.release()
cv2.destroyAllWindows()










