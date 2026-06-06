from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load model
model = load_model("model.h5")

# Change these labels if your training order is different
class_labels = [
    "pituitary",
    "glioma",
    "notumor",
    "meningioma"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['image']

    upload_path = os.path.join('static/uploads', file.filename)
    file.save(upload_path)

    img = load_img(upload_path, target_size=(128, 128))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    index = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)

    result = class_labels[index]

    return render_template(
        'index.html',
        result=result,
        confidence=round(confidence, 2),
        image_path=upload_path
    )

if __name__ == "__main__":
    app.run(debug=True)