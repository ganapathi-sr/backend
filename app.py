from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

app = Flask(__name__, static_folder="frontend/build", static_url_path="")

# Allow CORS only for your Netlify frontend
CORS(app, resources={r"/*": {"origins": "https://effervescent-scone-53bb1e.netlify.app"}})

# Define Model Path
MODEL_DIR = "backend/models"
MODEL_PATH = os.path.join(MODEL_DIR, "cotton_disease_model.h5")

# Ensure 'models' directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Google Drive File ID
file_id = "1ly8VuMzeXr7MDWLIqZnb1GVSszqTbFIH"

# Download model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", MODEL_PATH, quiet=False)

# Load Model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", str(e))
    model = None  # Prevent crashing if model fails to load

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
        print("❌ Error processing image:", str(e))
        return None

# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    image = request.files["file"]
    processed_image = preprocess_image(image)
    
    if processed_image is None:
        return jsonify({"error": "Invalid image"}), 400

    if model is None:
        return jsonify({"error": "Model failed to load"}), 500

    predictions = model.predict(processed_image)
    class_index = np.argmax(predictions)
    confidence = float(np.max(predictions))

    return jsonify({"disease": CLASS_LABELS[class_index], "confidence": confidence})

# Serve React Frontend
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react_app(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use environment variable for deployment
    app.run(debug=True, host="0.0.0.0", port=port)
