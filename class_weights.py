"""
y_train
   ↓
Check class distribution
   ↓
Calculate class weights
   ↓
Return class weights
   ↓
RNN/LSTM training
class_weights.py 
Handles class imbalance for the hate speech dataset. 
    - Checks class distribution 
    - Calculates class weights
"""
import numpy as np

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def print_class_distribution(y, name="Dataset"):
    """
    Display the class distribution.

    Parameters:
        y: Target labels.
        name: Name of the dataset.
    """

    print("\n" + "=" * 50)
    print(f"{name.upper()} CLASS DISTRIBUTION")
    print("=" * 50)

    classes, counts = np.unique(
        y,
        return_counts=True
    )

    total = len(y)

    for class_value, count in zip(classes, counts):
        percentage = (count / total) * 100

        print(
            f"Class {class_value}: "
            f"{count} samples "
            f"({percentage:.2f}%)"
        )


# ============================================================
# CALCULATE CLASS WEIGHTS
# ============================================================

def calculate_class_weights(y):
    """
    Calculate balanced class weights.

    Parameters:
        y: Training target labels.

    Returns:
        dict: Class weights.
    """

    classes = np.unique(y)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )

    class_weights = dict(
        zip(classes, weights)
    )

    print("\nCalculated class weights:")

    for class_value, weight in class_weights.items():
        print(
            f"Class {class_value}: "
            f"{weight:.4f}"
        )

    return class_weights
