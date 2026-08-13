"""
Dataset
   ↓
Preprocessing
   ↓
Tokenizer
   ↓
Padding
   ↓
Train/Test Split
   ↓
Class Weights
   ↓
   ┌───────────────┐
   │               │
   ▼               ▼
  RNN             LSTM
   │               │
   ▼               ▼
rnn_best.keras  lstm_best.keras
   │               │
   └───────┬───────┘
           ▼
   Model_Rvaluation.py
           │
           ▼
 Accuracy / Precision
 Recall / F1
 Confusion Matrix
 RNN vs LSTM
"""

"""
Model_Rvaluation.py

Evaluates the trained RNN and LSTM models.

Evaluation:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- RNN vs LSTM comparison
"""

import numpy as np

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from Preprocessed_Data import prepare_dataset


# ============================================================
# PATHS
# ============================================================

RNN_MODEL_PATH = "output_private/rnn_best.keras"
LSTM_MODEL_PATH = "output_private/lstm_best.keras"


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained model using the test dataset.
    """

    print("\n" + "=" * 60)
    print(f"{model_name.upper()} EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    probabilities = model.predict(
        X_test,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("Confusion Matrix:")
    print(cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        tokenizer,
        max_length
    ) = prepare_dataset()

    print("\nTest data loaded:")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # --------------------------------------------------------
    # Load RNN
    # --------------------------------------------------------

    print("\nLoading RNN model...")

    rnn_model = load_model(
        RNN_MODEL_PATH
    )

    # --------------------------------------------------------
    # Load LSTM
    # --------------------------------------------------------

    print("Loading LSTM model...")

    lstm_model = load_model(
        LSTM_MODEL_PATH
    )

    # --------------------------------------------------------
    # Evaluate RNN
    # --------------------------------------------------------

    rnn_results = evaluate_model(
        rnn_model,
        X_test,
        y_test,
        "RNN"
    )

    # --------------------------------------------------------
    # Evaluate LSTM
    # --------------------------------------------------------

    lstm_results = evaluate_model(
        lstm_model,
        X_test,
        y_test,
        "LSTM"
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RNN vs LSTM COMPARISON")
    print("=" * 60)

    print(
        f"\n{'Metric':<15}"
        f"{'RNN':>12}"
        f"{'LSTM':>12}"
    )

    print("-" * 39)

    print(
        f"{'Accuracy':<15}"
        f"{rnn_results['accuracy']:>12.4f}"
        f"{lstm_results['accuracy']:>12.4f}"
    )

    print(
        f"{'Precision':<15}"
        f"{rnn_results['precision']:>12.4f}"
        f"{lstm_results['precision']:>12.4f}"
    )

    print(
        f"{'Recall':<15}"
        f"{rnn_results['recall']:>12.4f}"
        f"{lstm_results['recall']:>12.4f}"
    )

    print(
        f"{'F1-Score':<15}"
        f"{rnn_results['f1_score']:>12.4f}"
        f"{lstm_results['f1_score']:>12.4f}"
    )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    if rnn_results["f1_score"] > lstm_results["f1_score"]:
        best_model = "RNN"
    else:
        best_model = "LSTM"

    print("\n" + "=" * 60)
    print(f"BEST MODEL: {best_model}")
    print("=" * 60)


if __name__ == "__main__":
    main()