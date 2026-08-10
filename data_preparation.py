"""
data_preparation.py

Handles:

- Creating a tokenizer
- Converting text into sequences
- Padding sequences to a fixed length
- Splitting data into training and testing sets
"""

import pickle

import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split


# ============================================================
# CREATE TOKENIZER
# ============================================================

def create_tokenizer(texts, num_words=None):
    """
    Create and fit a tokenizer on the text.

    Parameters:
        texts: Clean text data.
        num_words: Maximum vocabulary size.

    Returns:
        Tokenizer: Fitted Keras tokenizer.
    """

    tokenizer = Tokenizer(
        num_words=num_words,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(texts)

    print("\nTokenizer created successfully.")
    print(f"Vocabulary size: {len(tokenizer.word_index)}")

    return tokenizer


# ============================================================
# CONVERT TEXT TO SEQUENCES
# ============================================================

def text_to_sequences(tokenizer, texts):
    """
    Convert text into integer sequences.

    Example:
        "this is good"

        becomes something like:

        [15, 7, 23]
    """

    sequences = tokenizer.texts_to_sequences(texts)

    return sequences


# ============================================================
# PAD SEQUENCES
# ============================================================

def pad_text_sequences(sequences, max_length=None):
    """
    Pad sequences to a fixed length.

    Parameters:
        sequences: Integer sequences.
        max_length: Maximum sequence length.

    Returns:
        numpy.ndarray: Padded sequences.
    """

    if max_length is None:
        max_length = max(
            len(sequence)
            for sequence in sequences
        )

    padded_sequences = pad_sequences(
        sequences,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    print(f"\nMaximum sequence length: {max_length}")
    print(f"Padded data shape: {padded_sequences.shape}")

    return padded_sequences, max_length


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.

    Parameters:
        X: Input features.
        y: Target labels.
        test_size: Percentage used for testing.
        random_state: Reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print("\nData split completed.")

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    return X_train, X_test, y_train, y_test


# ============================================================
# COMPLETE TEXT REPRESENTATION PIPELINE
# ============================================================

def prepare_text_data(
    texts,
    labels,
    num_words=None,
    max_length=None,
    test_size=0.2,
    random_state=42
):
    """
    Complete text representation pipeline.

    Steps:
        1. Create tokenizer
        2. Convert text to sequences
        3. Pad sequences
        4. Split into training and testing data
    """

    # Create tokenizer
    tokenizer = create_tokenizer(
        texts,
        num_words=num_words
    )

    # Convert text to sequences
    sequences = text_to_sequences(
        tokenizer,
        texts
    )

    # Pad sequences
    X, max_length = pad_text_sequences(
        sequences,
        max_length
    )

    # Split data
    X_train, X_test, y_train, y_test = split_data(
        X,
        labels,
        test_size=test_size,
        random_state=random_state
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        tokenizer,
        max_length
    )


# ============================================================
# SAVE TOKENIZER
# ============================================================

def save_tokenizer(tokenizer, file_path="output_private/tokenizer.pkl"): 
    """ Save the tokenizer inside the output_private folder. """
    import os 
    import pickle 
    # Create output folder if it does not exist 
    os.makedirs("output_private", exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(tokenizer, file)

    print(f"\nTokenizer saved to {file_path}")

# ============================================================
# LOAD TOKENIZER
# ============================================================

def load_tokenizer(file_path):
    import pickle

    """
    Load the tokenizer from the output_private folder.

    Parameters:
        file_path: Path to the saved tokenizer file.
    """

    with open(file_path, "rb") as file:
        tok = pickle.load(file)

    print(f"\nTokenizer loaded from {file_path}")
    return tok