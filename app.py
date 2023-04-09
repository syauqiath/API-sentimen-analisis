import re
import numpy as np
import pandas as pd
from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from preprocessing import preprocessing_text, test_teks_nn, test_file_nn, test_teks_lstm, test_file_lstm

# ===== API =====
app = Flask(__name__)

app.json_encoder = LazyJSONEncoder
swagger_template = dict(
info = {
    'title': LazyString(lambda:'API Documentation for Sentiment Analysis'),
    'version': LazyString(lambda:'1.0.0'),
    'description': LazyString(lambda:'Dokumentasi API untuk Analisis Sentimen')
    },
    host = LazyString(lambda: request.host)
)
swagger_config = {
    "headers": [],
    "specs":[
        {
            "endpoint": "docs",
            "route": '/docs.json',
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger = Swagger(app, template=swagger_template,
                  config=swagger_config)



# ===== Tokenizer Neural Network =====
file = open("tknzr.pickle","rb")
tknzr = pickle.load(file)
file.close()

# ===== Load Model Neural Network =====
model_NN = load_model("model_nn/model_cnn.h5")

# ===== Load Model LSTM =====
model_LSTM = load_model('model_lstm/model_lstm.h5')



# ===== Homepage =====
@swag_from("docs/hello_world.yml", methods=['GET'])
@app.route('/', methods=['GET'])
def hello_world():
    json_response = {
        'status_code' : 200,
        'description' : "API untuk Analisis Sentimen : http://127.0.0.1:5000 ",
        'data'        : "Halo",
    }

    response_data = jsonify(json_response)
    return response_data


# ========== Neural Network ==========

# ===== Analisis dengan Neural Network (Teks) =====
@swag_from("docs/nn_text.yml", methods=['POST'])
@app.route('/nn-txt', methods=['POST'])
def nn_text():
    # Request Teks
    text = request.form.get('text')
    # Cleansing Teks
    text_clean = preprocessing_text(text)
    # Prediksi Sentimen 
    pred = test_teks_nn(text_clean)

    response_data = jsonify(pred)
    return response_data


# ===== Analisis dengan Neural Network (File) =====
@swag_from("docs/nn_file.yml", methods=['POST'])
@app.route('/nn-file', methods=['POST'])
def nn_file():
    # Input file
    file = request.files['file']
    # Get result from file in "List" format
    original_text = test_file_nn(file)

    response_data = jsonify(original_text)
    return response_data


# ========== LSTM ==========

# ===== Analisis dengan LSTM (Teks) =====
@swag_from("docs/lstm_text.yml", methods=['POST'])
@app.route('/lstm-text', methods=['POST'])
def lstm_text():
    # Request Teks
    text = request.form.get('text')
    # Cleansing Teks
    text_clean = preprocessing_text(text)
    # Prediksi Sentimen 
    pred = test_teks_lstm(text_clean)

    response_data = jsonify(pred)
    return response_data


# ===== Analisis dengan LSTM (File) =====
@swag_from("docs/lstm_file.yml", methods=['POST'])
@app.route('/lstm-file', methods=['POST'])
def lstm_file():
    # Input file
    file = request.files['file']
    # Get result from file in "List" format
    original_text = test_file_lstm(file)

    response_data = jsonify(original_text)
    return response_data


# ===== Menjalankan API =====
if __name__ == '__main__':
    app.run(debug=True)

# Default IP : http://127.0.0.1:5000