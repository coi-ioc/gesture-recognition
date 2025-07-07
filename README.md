# gesture-recognition
This project presents my Bachelor's thesis in Computer Engineering

# Riconoscimento Gesti con Deep Learning
Questo progetto di tesi si propone di realizzare un sistema per il riconoscimento di gesti della mano in tempo reale utilizzando una webcam, MediaPipe per l’estrazione dei landmark e una rete neurale sviluppata in PyTorch.

## 📂 Struttura del Progetto

```
gesture_recognition/
│
├── data/                          # File CSV dei dati gestuali raccolti
├── models/
│   └── model.py                   # Architettura della rete neurale
├── training/
│   └── train.py                   # Script di addestramento
├── gui/
│   └── webcam_handler.py          # Script per gestire webcam
├── inference/
│   └── predict.py                 # Script per l'inferenza
├── utils/
│   └── evaluation.py              # Funzione di valutazione
│   └── data_processing.py         # Caricamento e preprocessing dei dati
│   └── register_gesture.py        # Registrazione dei gesti tramite webcam
│   └── evaluation.py              # Stima di accuratezza
│   └── plot_confusion_matrix.py   # Stampa matrice di confusione per verica dati
├── app.py                         # Esecuzione in tempo reale con webcam
├── requirements.txt               # Dipendenze Python
└── README.md                      # Descrizione del progetto
```

## ⚙️ Risorse Utilizzate
- **Python 3.10**
- **PyTorch** – per il training della rete neurale
- **MediaPipe** – per l’estrazione dei landmark della mano
- **OpenCV** – per la gestione della webcam e visualizzazione
- **scikit-learn** – per preprocessing e encoding
- **NumPy**, **Pandas** – per la manipolazione dei dati

## 🙋‍♂️ Autore
Corrado Bruschi  
Tesi triennale in Ingegneria Elettronica ed Informatica – Università degli Studi di Pavia  
Anno accademico 2023/2024

