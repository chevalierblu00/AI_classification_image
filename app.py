import os
from flask import Flask, jsonify, render_template, request
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T

# Initialiser Flask
app = Flask(__name__)

# Charger le modèle
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.fc_stack(x)
        return x

num_classes = 100  # Exemple de nombre de classes
model = CNN(num_classes=num_classes)
dummy_input = torch.randn(1, 3, 28, 28)  # Example input size for the model
torch.onnx.export(model, dummy_input, "model.onnx", export_params=True)
model.eval()

# Définir les transformations pour les images
transform = T.Compose([
    T.Resize((28, 28)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Endpoint principal pour le traitement des images
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "Aucun fichier téléchargé"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Aucun fichier sélectionné"}), 400

        try:
            # Ouvrir et transformer l'image
            image = Image.open(file).convert("RGB")
            image_transformed = transform(image).unsqueeze(0)

            # Afficher des informations sur l'image transformée
            print(f"Taille de l'image après transformation : {image_transformed.shape}")
            print(f"Contenu des pixels : {image_transformed}")

            # Faire une prédiction
            with torch.no_grad():
                output = model(image_transformed)
                _, predicted = torch.max(output, 1)

            # Afficher les résultats du modèle
            print(f"Prédiction brute du modèle : {output}")
            print(f"Classe prédite : {predicted.item()}")

            # Retourner les données en JSON
            return jsonify({
                "predicted_class": predicted.item(),
                "raw_output": output.tolist(),  # Convertir les tenseurs en listes Python
                "image_pixels": image_transformed.tolist()  # Convertir les tenseurs en listes
            })
        except Exception as e:
            print(f"Erreur lors de la prédiction : {e}")
            return jsonify({"error": "Erreur lors de la prédiction"}), 500

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)