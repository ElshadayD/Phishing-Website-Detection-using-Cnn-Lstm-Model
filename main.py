from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

app = Flask(__name__)

# Load the pre-trained model
model = load_model('categorical_model.h5')

# Load the label encoder and tokenizer
le = LabelEncoder()
le.classes_ = np.load('label_encoder_classes.npy')
tokenizer = Tokenizer()
tokenizer.word_index = np.load('tokenizer_word_index.npy').item()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        url = request.form['url']
        # Tokenize and pad the input text
        seq = tokenizer.texts_to_sequences([url])
        padded = pad_sequences(seq, maxlen=100)
        # Make prediction
        prediction = np.argmax(model.predict(padded), axis=-1)[0]
        # Convert prediction back to original label
        predicted_label = le.inverse_transform([prediction])[0]
        return render_template('result.html', url=url, predicted_label=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
