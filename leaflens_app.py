# Imports
from flask import Flask, request, render_template, redirect
import os
import numpy as np
import tensorflow as tf
from keras import models
from keras.api.preprocessing import image

# Initiate Flask app object
app = Flask(__name__)

# Load pre-trained model (located in same directory as `leaflens_app.py`)
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'leaflens_model.keras')
leaflens_model = models.load_model(model_path)

# Define class names (in order of model training)
class_names = ['Apple - Apple Scab', 'Apple - Black Rot', 'Apple - Cedar Apple Rust', 'Apple - Healthy',
               'Blueberry - Healthy', 'Cherry (Including Sour) - Powdery Mildew', 'Cherry (Including Sour) - Healthy',
               'Corn (Maize) - Cercospora Leaf Spot Gray Leaf Spot', 'Corn (Maize) - Common Rust ',
               'Corn (Maize) - Northern Leaf Blight', 'Corn (Maize) - Healthy', 'Grape - Black Rot',
               'Grape - Esca (Black Measles)', 'Grape - Leaf Blight (Isariopsis Leaf Spot)', 'Grape - Healthy',
               'Orange - Haunglongbing (Citrus Greening)', 'Peach - Bacterial Spot', 'Peach - Healthy',
               'Pepper, Bell - Bacterial Spot', 'Pepper, Bell - Healthy', 'Potato - Early Blight',
               'Potato - Late Blight', 'Potato - Healthy', 'Raspberry - Healthy', 'Soybean - Healthy',
               'Squash - Powdery Mildew', 'Strawberry - Leaf Scorch', 'Strawberry - Healthy', 'Tomato - Bacterial Spot',
               'Tomato - Early Blight', 'Tomato - Late Blight', 'Tomato - Leaf Mold', 'Tomato - Septoria Leaf Spot',
               'Tomato - Spider Mites Two-Spotted Spider Mite', 'Tomato - Target Spot',
               'Tomato - Tomato Yellow Leaf Curl Virus', 'Tomato - Tomato Mosaic Virus', 'Tomato - Healthy']


def model_predict(img_path, model):
    """Takes an image file and a pre-defined model as input, returns prediction and associated probability."""
    # Load image and convert to array
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    # Add batch dimension
    x = np.expand_dims(x, axis=0)
    # Get prediction from model
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    """Saves user-selected image locally, then calls model_predict with image and leaflens_model"""
    # Check if file present in POST
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Save uploaded image locally, then send to model_predict to return result
    if file:
        # Check `uploads` folder exists, create if not:
        if not os.path.isdir('uploads'):
            os.makedirs('uploads')

        # Save uploaded image locally
        file_path = os.path.join(base_path, 'uploads', file.filename)
        file.save(file_path)

        # Call model_predict to obtain results and return to user
        preds = model_predict(file_path, leaflens_model)
        probs = tf.nn.softmax(preds).numpy()

        result = class_names[np.argmax(probs)]
        prob = np.max(probs)

        return f"Leaf identified as {result}\nwith {prob*100:.2f}% probability."
    return None


if __name__ == '__main__':
    app.run(debug=True)
