import numpy as np 
import tensorflow as tf 
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st


# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

model = load_model('simple_rnn_imdb.h5')


def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review):
    processed = preprocess_text(review)
    prediction = model.predict(processed)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]


##streamlit app

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter the move review to classsify it as positive or neagative')


user_input = st.text_area('Movie Review')

if st.button('Classify'):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a movie review.")
    else:
        sentiment, score = predict_sentiment(user_input)
        st.write(f'**Sentiment**: {sentiment}')
        st.write(f'**Prediction Score**: {score:.4f}')
else:
    st.info('Please enter a movie review and click **Classify**.')