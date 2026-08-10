"""
Dataset CSV
    ↓
Preprocessed_Data.py
    │
    ├── preprocess.py
    │      ├── Load dataset
    │      ├── Explore
    │      └── Clean text
    │
    └── data_preparation.py
           ├── Tokenizer
           ├── Sequences
           ├── Padding
           └── Train/Test split
    ↓
Prepared data
    ↓
output_private/
"""
import os

from preprocess import (
    read_dataset,
    print_data_info,
    clean_dataset,
    preprocess_data
)

from data_preparation import (
    prepare_text_data,
    save_tokenizer
)

# ============================================================
# PATHS
# ============================================================

DATASET_PATH = os.path.join(
    "dataset",
    "Dataset---Hate-Speech-Detection-using-Deep-Learning.csv"
)

TOKENIZER_PATH = os.path.join(
    "output_private",
    "tokenizer.pkl"
)


# ============================================================
# MAIN PIPELINE
# ============================================================

def prepare_dataset():
    """
    Run the complete preprocessing and text representation
    pipeline.
    """

    print("\n" + "=" * 60)
    print("STARTING DATA PREPARATION")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    data = read_dataset(DATASET_PATH)

    # --------------------------------------------------------
    # 2. Explore dataset
    # --------------------------------------------------------

    print_data_info(data)

    # --------------------------------------------------------
    # 3. Clean dataset
    # --------------------------------------------------------

    data = clean_dataset(data)

    # --------------------------------------------------------
    # 4. Text preprocessing
    # --------------------------------------------------------

    data = preprocess_data(data)

    # --------------------------------------------------------
    # 5-8. Text representation
    # --------------------------------------------------------

    X = data["clean_text"]
    y = data["class"]

    (
        X_train,
        X_test,
        y_train,
        y_test,
        tokenizer,
        max_length
    ) = prepare_text_data(
        texts=X,
        labels=y,
        num_words=None,
        max_length=None,
        test_size=0.2,
        random_state=42
    )

    # --------------------------------------------------------
    # 9. Save tokenizer
    # --------------------------------------------------------

    os.makedirs(
        "output_private",
        exist_ok=True
    )

    save_tokenizer(
        tokenizer,
        TOKENIZER_PATH
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED")
    print("=" * 60)

    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")

    print(f"\ny_train shape: {y_train.shape}")
    print(f"y_test shape:  {y_test.shape}")

    print(f"\nNumber of classes: {y.nunique()}")
    print(f"Classes: {sorted(y.unique())}")

    print(f"\nMaximum sequence length: {max_length}")

    print(f"\nTokenizer saved to:")
    print(TOKENIZER_PATH)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        tokenizer,
        max_length
    )

