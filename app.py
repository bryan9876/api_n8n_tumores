import os
import requests

def descargar_modelo():
    modelo_path = "modelo_tumor.h5"
    if not os.path.exists(modelo_path):
        print("📥 Descargando modelo desde Google Drive...")
        url = "https://drive.google.com/uc?export=download&id=1pyC6WqNNS6NlIIzdAl-bs2MzqR4bgw3T"
        r = requests.get(url, allow_redirects=True)
        with open(modelo_path, 'wb') as f:
            f.write(r.content)
        print("✅ Modelo descargado.")
    else:
        print("✅ Modelo ya existe.")

descargar_modelo()

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Cargar el modelo
modelo = load_model("modelo_tumor.h5")

# Ruta base
@app.route("/", methods=["GET"])
def home():
    return "✅ API para detección de tumores cerebrales"

# Ruta de predicción
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No se recibió ninguna imagen"}), 400
    
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Realizar predicción
    pred = modelo.predict(img_array)[0][0]
    response = {
        "tumor_detectado": bool(pred > 0.5),
        "confianza": round(float(pred if pred > 0.5 else 1 - pred), 4)
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
