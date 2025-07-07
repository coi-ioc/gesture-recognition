import torch
import joblib
from models.model import GestureRecognitionModel

def predict(input_data, model_path, scaler_encoder_path):
    # Carica scaler ed encoder
    try:
        scaler, encoder = joblib.load(scaler_encoder_path)
    except Exception as e:
        print(f"Errore nel caricamento di scaler ed encoder: {e}")
        return None

    # Verifica che encoder sia stato caricato correttamente
    if not hasattr(encoder, 'classes_'):
        print("Encoder non contiene 'classes_'")
        return None

    # Calcola il numero di classi dopo aver caricato l'encoder
    num_classes = len(encoder.classes_)

    # Carica il modello
    input_size = input_data.shape[1]
    model = GestureRecognitionModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Preprocessa i dati
    input_data = scaler.transform(input_data)
    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    # Predizione
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_class = torch.max(outputs, 1)
    predicted_label = encoder.inverse_transform([predicted_class.item()])[0]  # Restituisce la stringa
    return str(predicted_label)






    
