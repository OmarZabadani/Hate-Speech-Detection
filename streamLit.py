"""
Streamlit application for Hate Speech Detection.

Uses:
    - Saved LSTM model
    - Saved tokenizer
    - Existing text preprocessing
"""

import os
import pickle

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocess import preprocess_text


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = os.path.join(
    "output_private",
    "lstm_best.keras"
)

TOKENIZER_PATH = os.path.join(
    "output_private",
    "tokenizer.pkl"
)

MAX_LENGTH = 29


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_lstm_model():
    return load_model(MODEL_PATH)


# ============================================================
# LOAD TOKENIZER
# ============================================================

@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as file:
        return pickle.load(file)


# ============================================================
# PREDICTION
# ============================================================

def predict_text(text, model, tokenizer):
    """
    Preprocess text and predict its class.
    """

    # Use the same preprocessing used during training
    cleaned_text = preprocess_text(text)

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences(
        [cleaned_text]
    )

    # Pad sequence
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    # Predict
    probabilities = model.predict(
        padded_sequence,
        verbose=0
    )

    predicted_class = int(
        np.argmax(probabilities[0])
    )

    confidence = float(
        probabilities[0][predicted_class]
    )

    return predicted_class, confidence, cleaned_text


# ============================================================
# STREAMLIT INTERFACE
# ============================================================

st.set_page_config(
    page_title="Hate Speech Detection",
    layout="centered"
)


st.title("Hate Speech Detection")

st.write(
    "Enter a text below and the trained LSTM model "
    "will predict its class."
)


# ============================================================
# LOAD RESOURCES
# ============================================================

try:
    model = load_lstm_model()
    tokenizer = load_tokenizer()

except Exception as error:
    st.error(
        f"Failed to load model or tokenizer:\n\n{error}"
    )
    st.stop()


# ============================================================
# TEXT INPUT
# ============================================================

text = st.text_area(
    "Enter text:",
    placeholder="Type a tweet or message here...",
    height=150
)


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button(
    "Predict",
    type="primary"
):

    if not text.strip():

        st.warning(
            "Please enter some text first."
        )

    else:

        predicted_class, confidence, cleaned_text = (
            predict_text(
                text,
                model,
                tokenizer
            )
        )

        st.subheader("Prediction")

        st.write(
            f"**Predicted Class:** {predicted_class}"
        )

        st.write(
            f"**Confidence:** {confidence:.2%}"
        )

        with st.expander("View processed text"):
            st.write(cleaned_text)