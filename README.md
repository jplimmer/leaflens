# LeafLens

**LeafLens** is a web-based Python application that checks if a leaf displays signs of certain crop diseases.

To predict the plant's health, the app uses a Sequential Keras model (see [pre-requisites](#pre-requisites)) that has been trained on the [Plant Village](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset/) dataset. This is a dataset of over 54,000 images of healthy and unhealthy leaves from 14 species of crop.

This was a short project primarily aimed at getting comfortable working with Big Data and image classification in TensorFlow.


## Tech Stack
- Backend: Python, Flask
- Predictive model: TensorFlow/Keras
- Frontend: HTML, CSS (via Flask's templating engine)


## Usage
<img src="images/leaflens_demo.gif" width="70%">

When the application (python file) is running locally, LeafLens can be used at http://127.0.0.1:5000.

On the homepage, the user clicks `Choose File` and selects an image from their directory.

When the user clicks `Check Health`, the application saves a copy of the image in the repository 'uploads' folder and passes the image to the trained model, and then the model prediction and associated proability are displayed in the app.


## Pre-requisites
- Python 3.x
- Flask 2.x
- Tensorflow 2.14+
- Keras 3.4+

The predictive Keras model can be compiled and saved in your Google Drive using the [Google Colab notebook](leaflens_model_training.ipynb). It then needs to be copied locally to the same directory as `leaflens_app.py`.


## Future Development Ideas
- Convert to mobile application for ease-of-use with camera.
- Improve model training scores on existing dataset.


