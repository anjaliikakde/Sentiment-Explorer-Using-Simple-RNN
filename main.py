import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

# Helper Functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Streamlit app
st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬", layout="wide")
st.markdown("<h1 style='text-align: center; color: #FF6347;'>IMDB Movie Review Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Enter a movie review to classify it as positive or negative.</h4>", unsafe_allow_html=True)

# User input
user_input = st.text_area('Movie Review', height=200, placeholder='Type your movie review here...')

if st.button('Classify', help="Click to classify the review as positive or negative"):
    if user_input:
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
        st.markdown(f"<h3 style='text-align: center;'>Sentiment: {sentiment}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>Prediction Score: {prediction[0][0]:.4f}</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='text-align: center; color: red;'>Please enter a movie review.</p>", unsafe_allow_html=True)

st.markdown("<footer style='text-align: center; color: gray;'>Built with ❤ using Streamlit</footer>", unsafe_allow_html=True)