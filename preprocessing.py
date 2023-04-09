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

#Fungsi cleansing data 
def preprocessing_text(text):
    text = text.strip()
    text = text.lower()
    text = text.replace("\\n"," ")
    text = re.sub(r"(\s)(\1+)",r"\1",text)
    text = text.replace("rt","")
    text = text.replace("user ","")
    text = text.replace(" user","")
    text = re.sub(r"([a-z])(\1{3,})",r"\1\1",text)
    text = re.sub(r"(\\x)([a-z0-9]{2})",r"",text)
    text = text.replace("\\x8","")
    text = text.strip()
    
    return text

#Fungsi untuk inputan kalimat 
max_features=10000

file = open("tknzr.pickle","rb")
tknzr = pickle.load(file)
file.close()

# ===== Neural Network =====

# load model
from keras.models import load_model
model_NN = load_model("model_nn\model_cnn.h5")

# teks nn
def test_teks_nn(kalimat):
    input_kalimat = [kalimat]
    input_kalimat = tknzr.texts_to_sequences(input_kalimat)
    input_kalimat = pad_sequences(input_kalimat, maxlen=64)
    
    hasil = model_NN.predict(input_kalimat)
    hasil = hasil.argmax(axis=1)

    
    # konversi nilai prediksi menjadi label sentimen
    labels = {0: "negative", 1: "neutral", 2: "positive"}
    hasil = labels[hasil[0]]
    return hasil

# file nn
def test_file_nn(file):
    df = pd.read_csv(file,header=0, encoding='latin-1')
    df['text_clean'] = df.apply(lambda row : preprocessing_text(row['tweets']), axis = 1)
    input_kalimat = df['text_clean'].to_list()
    df.drop(columns=['labels'], inplace=True)
    input_kalimat = tknzr.texts_to_sequences(input_kalimat)
    input_kalimat = pad_sequences(input_kalimat, maxlen=64)
    
    hasil = model_NN.predict(input_kalimat)
    hasil = hasil.argmax(axis=-1)

    
    # konversi nilai prediksi menjadi label sentimen
    labels = {0: "negative", 1: "neutral", 2: "positive"}
    df["label_prediksi"] = [labels[pred] for pred in hasil.tolist()]
    df = df.to_dict(orient="records")
    return df


# ===== LSTM =====

# load model
from keras.models import load_model
model_LSTM = load_model("model_lstm\model_lstm.h5")

# teks lstm
def test_teks_lstm(kalimat):
    input_kalimat = [kalimat]
    input_kalimat = tknzr.texts_to_sequences(input_kalimat)
    input_kalimat = pad_sequences(input_kalimat, maxlen=64)
    
    hasil = model_LSTM.predict(input_kalimat)
    hasil = hasil.argmax(axis=1)

    
    # konversi nilai prediksi menjadi label sentimen
    labels = {0: "negative", 1: "neutral", 2: "positive"}
    hasil = labels[hasil[0]]
    return hasil

# file lstm
def test_file_lstm(file):
    df = pd.read_csv(file,header=0, encoding='latin-1')
    df['text_clean'] = df.apply(lambda row : preprocessing_text(row['tweets']), axis = 1)
    input_kalimat = df['text_clean'].to_list()
    df.drop(columns=['labels'], inplace=True)
    input_kalimat = tknzr.texts_to_sequences(input_kalimat)
    input_kalimat = pad_sequences(input_kalimat, maxlen=64)
    
    hasil = model_LSTM.predict(input_kalimat)
    hasil = hasil.argmax(axis=-1)

    
    # konversi nilai prediksi menjadi label sentimen
    labels = {0: "negative", 1: "neutral", 2: "positive"}
    df["label_prediksi"] = [labels[pred] for pred in hasil.tolist()]
    df = df.to_dict(orient="records")
    return df