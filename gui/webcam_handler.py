import cv2
import numpy as np
from inference.predict import predict
import torch
from sklearn.preprocessing import LabelEncoder
import joblib

class WebcamHandler:
    def __init__(self, update_text_callback):
        self.update_text_callback = update_text_callback
        self.model_path = "gesture_recognition_model.pth"
        self.scaler = joblib.load("scaler.pkl")  # Carica lo scaler salvato
        self.encoder = joblib.load("encoder.pkl")  # Carica l'encoder salvato
        self.start_webcam()

    def start_webcam(self):
        # Inizializza la webcam
        self.cap = cv2.VideoCapture(0)  # Usa la prima webcam disponibile
        if not self.cap.isOpened():
            print("Errore nell'aprire la webcam.")
            return

        while True:
            # Acquisisci un frame dalla webcam
            ret, frame = self.cap.read()
            if not ret:
                break

            # Pre-elaborazione del frame (ad esempio ridimensionamento, conversione in grigio, etc.)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Assicurati che la tua pre-elaborazione sia conforme al formato di input del modello

            # Estrai i keypoints della mano destra (usa MediaPipe o simili)
            input_data = np.random.rand(1, 63)  # Sostituisci con la logica corretta per estrarre caratteristiche

            # Fai la previsione
            predicted_label = predict(input_data, self.model_path, self.scaler, self.encoder)
            
            # Mostra il risultato sulla finestra
            cv2.putText(frame, f"Gesto: {predicted_label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Mostra il frame con il gesto previsto
            cv2.imshow('Webcam - Riconoscimento Gesti', frame)

            # Interrompi se premi il tasto 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Rilascia la webcam e chiudi le finestre
        self.cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        """Ferma la webcam."""
        self.cap.release()
        cv2.destroyAllWindows()




