# Imports
from flask import Flask, request, render_template, redirect, url_for
import os
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, preprocessing
from preprocessing import image

path = 'C:\\Users\\james\\OneDrive\\Documents\\Coding\\Nod Bootcamp\\Projects\\Final Project\\'

# Initiate Flask app object
app = Flask(__name__)

# Load your pre-trained model
model = models.load_model(path+'plant_model.keras')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))  # Change size as per your model's input size
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', file.filename)
        file.save(file_path)

        preds = model_predict(file_path, model)
        result = np.argmax(preds)  # Modify as per your model's output

        return str(result)
    return None


if __name__ == '__main__':
    app.run(debug=True)

