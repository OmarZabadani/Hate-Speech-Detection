
"""
preprocess.py

Handles:
- Loading the dataset
- Exploring the dataset
- Checking target classes
- Checking class distribution
- Removing duplicates and missing values
- Text preprocessing:
    - Lowercase
    - Remove URLs
    - Remove punctuation
    - Remove numbers
    - Remove stopwords
    - Tokenization
    - Lemmatization
"""

import pandas as pd
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer




# NLP tools
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ============================================================
# READ DATASET
# ============================================================

def read_dataset(file_path):
    """
    Load the dataset into a pandas DataFrame.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: Loaded dataset.
    """

    return pd.read_csv(file_path)


# ============================================================
# DATASET INFORMATION
# ============================================================

def print_data_info(data):
    """
    Display basic information about the dataset.
    """

    print("\n========== DATA INFO ==========")

    print("\nDataset shape:")
    print(data.shape)

    print("\nColumns:")
    print(data.columns.tolist())

    print("\nFirst 5 rows:")
    print(data.head())

    print("\nData types:")
    print(data.dtypes)

    print("\nMissing values:")
    print(data.isnull().sum())

    print("\nDuplicate rows:")
    print(data.duplicated().sum())

    print("\nClass distribution:")
    print(data["class"].value_counts().sort_index())

    print("\nClass distribution (%):")
    print(
        data["class"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )


# ============================================================
# CLEAN DATASET
# ============================================================

def clean_dataset(data):
    """
    Remove missing values and duplicate rows.
    """

    print("\n========== DATA CLEANING ==========")

    before = len(data)

    # Remove rows with missing values
    data = data.dropna(subset=["tweet", "class"])

    # Remove duplicate rows
    data = data.drop_duplicates()

    # Make sure tweet values are strings
    data["tweet"] = data["tweet"].astype(str)

    after = len(data)

    print(f"Rows before cleaning: {before}")
    print(f"Rows after cleaning:  {after}")
    print(f"Rows removed:        {before - after}")

    return data.reset_index(drop=True)


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):
    """
    Clean a single tweet.

    Steps:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove mentions
    4. Remove numbers
    5. Remove punctuation
    6. Tokenize
    7. Remove stopwords
    8. Lemmatize
    """

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove Twitter mentions
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # Remove numbers
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Remove extra whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [
        word
        for word in tokens
        if word not in STOP_WORDS
    ]

    # Lemmatization
    tokens = [
        LEMMATIZER.lemmatize(word)
        for word in tokens
    ]

    # Convert tokens back into text
    return " ".join(tokens)


# ============================================================
# APPLY PREPROCESSING TO DATASET
# ============================================================

def preprocess_data(data):
    """
    Apply text preprocessing to all tweets.
    """

    data = data.copy()

    print("\n========== TEXT PREPROCESSING ==========")

    data["clean_text"] = data["tweet"].apply(
        preprocess_text
    )

    # Remove empty texts
    data = data[
        data["clean_text"].str.strip() != ""
    ]

    data = data.reset_index(drop=True)

    print("Text preprocessing completed.")

    print("\nOriginal vs cleaned tweets:")

    print(
        data[
            ["tweet", "clean_text"]
        ]
        .head(10)
        .to_string(index=False)
    )

    return data

