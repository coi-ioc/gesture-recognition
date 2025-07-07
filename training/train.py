import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.model import GestureRecognitionModel
from data_processing import load_and_preprocess_data
import pandas as pd
import os
import pickle


def train_model(data_dir, epochs=100, batch_size=32, learning_rate=0.001):
    # Carica i dati
    X, y, scaler, encoder = load_and_preprocess_data(data_dir)

    # Suddividi i dati in training e test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Converti i dati in tensori PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Inizializza il modello
    input_size = X_train.shape[1]
    num_classes = len(set(y))
    model = GestureRecognitionModel(input_size, num_classes)
    print("Numero di feature durante l'addestramento:", X_train.shape[1])




    # Definisci la funzione di perdita e l'ottimizzatore
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Addestramento
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # Valutazione
    from utils.evaluation import evaluate_model
    accuracy = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Salva il modello
    torch.save(model.state_dict(), "gesture_recognition_model.pth")
    print("Modello salvato!")

    # Salva scaler ed encoder
    with open("scaler_encoder.pkl", "wb") as f:
        pickle.dump((scaler, encoder), f)
    print("Scaler ed encoder salvati!")

    class_names = encoder.classes_
    return model, scaler, encoder, test_loader, class_names

