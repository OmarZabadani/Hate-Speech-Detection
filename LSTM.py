"""
LSTM.py

Handles:
- Building the LSTM model
- Training the LSTM model
- Saving the trained LSTM model
"""

import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# BUILD LSTM MODEL
# ============================================================

def build_lstm_model(
    vocab_size,
    max_length,
    embedding_dim=128,
    lstm_units=64,
    num_classes=3
):
    """
    Build and compile the LSTM model.

    Parameters:
        vocab_size: Number of words in the tokenizer vocabulary.
        max_length: Maximum sequence length.
        embedding_dim: Embedding vector size.
        lstm_units: Number of LSTM units.
        num_classes: Number of target classes.

    Returns:
        Compiled LSTM model.
    """

    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length
        ),

        LSTM(
            lstm_units,
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
# TRAIN LSTM MODEL
# ============================================================

def train_lstm_model(
    model,
    X_train,
    y_train,
    class_weights=None,
    epochs=10,
    batch_size=32,
    validation_split=0.2
):
    """
    Train the LSTM model.

    Parameters:
        model: Compiled LSTM model.
        X_train: Training sequences.
        y_train: Training labels.
        class_weights: Class weights for imbalanced data.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        validation_split: Validation data percentage.

    Returns:
        Training history.
    """

    os.makedirs(
        "output_private",
        exist_ok=True
    )

    checkpoint = ModelCheckpoint(
        "output_private/lstm_best.keras",
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
# SAVE LSTM MODEL
# ============================================================

def save_lstm_model(
    model,
    file_path="output_private/lstm_model.keras"
):
    """
    Save the trained LSTM model.
    """

    os.makedirs(
        "output_private",
        exist_ok=True
    )

    model.save(file_path)

    print(
        f"\nLSTM model saved to: {file_path}"
    )