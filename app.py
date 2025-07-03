from flask import Flask, request, render_template, redirect, url_for, flash
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import json
from werkzeug.utils import secure_filename

# Flask app initialization
app = Flask(__name__, template_folder='.')  # HTML files are in root directory
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "fallback_dev_key")

# Enable CORS for all domains (for frontend-backend communication)
CORS(app)

# Upload folder configuration
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure upload folder exists

# Model configuration
IMAGE_SIZE = 224

# Properly resolve absolute paths for model and label map
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "rice_model.keras")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "model", "label_map.json")

# Load trained model
model = load_model(MODEL_PATH)

# Load label map
with open(LABEL_MAP_PATH, 'r') as f:
    label_map = json.load(f)

# Convert label_map (dict) into sorted list
CATEGORIES = [label_map[str(i)] for i in range(len(label_map))]

# Image preprocessing and prediction
def model_predict(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or corrupted.")
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0]
    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    predicted_label = CATEGORIES[class_index]
    return predicted_label, confidence

# Route: Home
@app.route('/')
def home():
    return render_template('index.html')

# Route: Upload/Prediction Page
@app.route('/details')
def details():
    return render_template('details.html')

# Route: Results Display Page
@app.route('/results')
def results():
    rice_type = request.args.get('type', 'Unknown')
    confidence = request.args.get('confidence', '0')
    image_url = request.args.get('image_url', '')
    return render_template('results.html', type=rice_type, confidence=confidence, image_url=image_url)

# Route: Prediction Logic
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400
    file = request.files['image']
    if file.filename == '':
        return "Empty filename", 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    try:
        label, confidence = model_predict(filepath)
        confidence_percent = round(confidence * 100, 2)
        image_url = f"/static/uploads/{filename}"
        return redirect(url_for('results', type=label, confidence=confidence_percent, image_url=image_url))
    except Exception as e:
        print("Prediction Error:", e)
        return "Prediction failed. Please check the server logs.", 500

# Route: Contact form handling
@app.route('/contact', methods=['POST'])
def contact():
    name = request.form.get('name')
    email = request.form.get('email')
    subject = request.form.get('subject')
    message = request.form.get('message')

    # Save message to text file
    with open("messages.txt", "a", encoding="utf-8") as f:
        f.write(f"Name: {name}\nEmail: {email}\nSubject: {subject}\nMessage: {message}\n---\n")

    print(f"New Contact Form Submission:\nName: {name}\nEmail: {email}\nSubject: {subject}\nMessage: {message}\n")

    flash("✅ Your message has been sent successfully!", "success")
    return redirect(url_for('home'))

# Local run (development only)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
