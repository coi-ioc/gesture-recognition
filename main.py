from training.train import train_model
from inference.predict import predict
from utils.plot_confusion_matrix import plot_confusion_matrix
import numpy as np

if __name__ == "__main__":
    # Addestra il modello
    data_dir = "data/"
    model, scaler, encoder, test_loader, class_names = train_model(data_dir)

    plot_confusion_matrix(model, test_loader, class_names)

    # Esempio di inferenza
    input_data = np.random.rand(1, 63)  
    predicted_label = predict(input_data, "gesture_recognition_model.pth", "scaler_encoder.pkl")
    print("Predicted Gesture:", predicted_label)



