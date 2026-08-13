"""
RNN.py

Handles:
- Building the Simple RNN model
- Training the Simple RNN model
- Saving the trained model
"""

import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# BUILD RNN MODEL
# ============================================================

def build_rnn_model(
    vocab_size,
    max_length,
    embedding_dim=128,
    rnn_units=64,
    num_classes=3
):
    """
    Build a Simple RNN model.

    Parameters:
        vocab_size: Number of words in the tokenizer vocabulary.
        max_length: Maximum sequence length.
        embedding_dim: Size of word embeddings.
        rnn_units: Number of RNN units.
        num_classes: Number of target classes.

    Returns:
        Compiled Keras RNN model.
    """

    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length
        ),

        SimpleRNN(
            rnn_units,
            return_sequences=False
        ),

        Dropout(0.3),

        Dense(
            64,
            activation="relu"
        ),

        Dropout(0.3),

        Dense(
            num_classes,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# TRAIN RNN MODEL
# ============================================================

def train_rnn_model(
    model,
    X_train,
    y_train,
    class_weights=None,
    epochs=10,
    batch_size=32,
    validation_split=0.2
):
    """
    Train the Simple RNN model.

    Parameters:
        model: Compiled RNN model.
        X_train: Training sequences.
        y_train: Training labels.
        class_weights: Optional class weights.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        validation_split: Portion of training data used for validation.

    Returns:
        Training history.
    """

    os.makedirs(
        "output_private",
        exist_ok=True
    )

    checkpoint = ModelCheckpoint(
        "output_private/rnn_best.keras",
        monitor="val_loss",
        save_best_only=True
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[
            checkpoint,
            early_stopping
        ],
        verbose=1
    )

    return history


# ============================================================
# SAVE RNN MODEL
# ============================================================

def save_rnn_model(
    model,
    file_path="output_private/rnn_model.keras"
):
    """
    Save the trained RNN model.
    """

    os.makedirs(
        "output_private",
        exist_ok=True
    )

    model.save(file_path)

    print(
        f"\nRNN model saved to: {file_path}"
    )