from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown

app = Flask(__name__, static_folder="frontend/build", static_url_path="")
CORS(app)  # Enable CORS for frontend communication

# Model Path
MODEL_DIR = "backend/models"
MODEL_PATH = os.path.join(MODEL_DIR, "cotton_disease_model.h5")
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "cotton_disease_model.tflite")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Google Drive Model File ID
file_id = "1ly8VuMzeXr7MDWLIqZnb1GVSszqTbFIH"

# Download Model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", MODEL_PATH, quiet=False)

# Lazy Model Loading
model = None  # Load model only when needed

def get_model():
    global model
    if model is None:
        print("Loading model into memory...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    return model

# Class Labels
CLASS_LABELS = ["Aphids", "Army Worm", "Bacterial Blight", "Healthy", "Powdery Mildew", "Target Spot"]

# Preprocess Image
def preprocess_image(image):
    try:
        image = Image.open(image).convert("RGB").resize((224, 224))
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        print("Error processing image:", str(e))
        return None

# Handle CORS Preflight Requests
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        response = jsonify({"message": "CORS Preflight OK"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        return response, 200

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    image = request.files["file"]
    processed_image = preprocess_image(image)

    if processed_image is None:
        return jsonify({"error": "Invalid image"}), 400

    model = get_model()  # Load model only when needed
    predictions = model.predict(processed_image)
    class_index = np.argmax(predictions)
    confidence = float(np.max(predictions))

    response = jsonify({"disease": CLASS_LABELS[class_index], "confidence": confidence})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

# Serve React Frontend
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react_app(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
