import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_and_preprocess_data(data_dir="C:/Users/ROG/Desktop/gesture_recognition/data"):
    # Verifica il percorso completo della cartella
    print(f"Caricamento dati dalla cartella: {data_dir}")
    
    # Controlla se la cartella esiste
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"La cartella specificata non esiste: {data_dir}")
    
    # Carica i file CSV
    dataframes = []
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(data_dir, file)
            print(f"Caricamento del file: {file_path}")
            try:
                df = pd.read_csv(file_path, header=None)
                # Verifica il numero di colonne
                if df.shape[1] == 64:  # 63 per i landmark + 1 per l'etichetta
                    dataframes.append(df)
                else:
                    print(f"File {file} ignorato: ha {df.shape[1]} colonne invece di 64.")
            except Exception as e:
                print(f"Errore nel caricamento del file {file}: {e}")
    
    # Controlla se sono stati caricati dati validi
    if not dataframes:
        raise ValueError("Nessun file valido trovato nella cartella specificata.")
    
    # Combina i dati
    data = pd.concat(dataframes, ignore_index=True)
    print(f"Dimensioni totali dei dati combinati: {data.shape}")
    
    # Rimuovi righe duplicate e valori mancanti
    data = data.drop_duplicates()
    data = data.dropna()
    print(f"Dimensioni dopo rimozione di duplicati e valori mancanti: {data.shape}")
    
    # Controlla valori non numerici e sostituiscili con 0
    if not data.apply(lambda x: np.isreal(x).all(), axis=1).all():
        print("Valori non numerici rilevati nei dati, sostituzione con 0.")
        data = data.apply(pd.to_numeric, errors="coerce").fillna(0)
    
    # Estrai feature (X) e label (y)
    X = data.iloc[:, 1:].values  # Tutte le colonne tranne la prima
    y = data.iloc[:, 0].values   # La prima colonna è l'etichetta
    
    # Controlla dimensioni di X e y
    print(f"Dimensioni di X (feature): {X.shape}")
    print(f"Dimensioni di y (etichette): {y.shape}")
    
    # Normalizza i dati
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    # Codifica le etichette
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    # Controlla le classi uniche
    unique_classes = set(y_encoded)
    print(f"Classi uniche trovate: {unique_classes}")
    if len(unique_classes) == 1:
        print("Attenzione: tutte le etichette sono uguali. Verifica i file CSV.")
    
    return X, y_encoded, scaler, encoder



