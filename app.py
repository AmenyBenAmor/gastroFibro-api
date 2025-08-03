from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import json
app = Flask(__name__)

MODEL_PATH = "model/best_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)



with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Inverser le mapping
idx_to_class = {v: k for k, v in class_indices.items()}

def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0]
    predicted_index = int(np.argmax(prediction))
    predicted_label = idx_to_class[predicted_index]

    # Liste des classes considérées comme maladies
    maladies = [
        "Colorectal cancer",
        "Esophagitis",
        "Gastric polyps",
        "Mucosal inflammation large bowel",
        "Ulcer"
    ]

    if predicted_label in maladies:
        return f"malade : {predicted_label}"
    else:
        return "sain"


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join("temp_image.jpg")
    file.save(file_path)

    try:
        result = predict_image(file_path)
        os.remove(file_path)
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Main ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
