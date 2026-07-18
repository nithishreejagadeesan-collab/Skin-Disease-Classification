from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("model/skin_disease_model.h5")

# HAM10000 Classes
classes = [
    "Actinic Keratoses (akiec)",
    "Basal Cell Carcinoma (bcc)",
    "Benign Keratosis (bkl)",
    "Dermatofibroma (df)",
    "Melanoma (mel)",
    "Melanocytic Nevus (nv)",
    "Vascular Lesion (vasc)"
]

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No image selected"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    print("Saved File:", filepath)
    print("File Size:", os.path.getsize(filepath))

    try:
        img = Image.open(filepath).convert("RGB")
    except UnidentifiedImageError:
        return "❌ Uploaded file is not a valid image."

    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    class_index = np.argmax(prediction)
    result = classes[class_index]
    confidence = round(float(np.max(prediction)) * 100, 2)

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        image=filepath
    )


if __name__ == "__main__":
    app.run(debug=True)