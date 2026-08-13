"""
Preprocessed_Data.py
        ↓
prepare_dataset()
        ↓
y_train
        ↓
class_weights.py
        ↓
calculate_class_weights()
        ↓
LSTM.py
        ↓
build_lstm_model()
        ↓
train_lstm_model()
"""
from Preprocessed_Data import prepare_dataset
from class_weights import calculate_class_weights

from LSTM import (
    build_lstm_model,
    train_lstm_model
)


# ============================================================
# 1. PREPARE DATA
# ============================================================

(
    X_train,
    X_test,
    y_train,
    y_test,
    tokenizer,
    max_length
) = prepare_dataset()


# ============================================================
# 2. CALCULATE CLASS WEIGHTS
# ============================================================

class_weights = calculate_class_weights(y_train)


# ============================================================
# 3. MODEL INFORMATION
# ============================================================

vocab_size = len(tokenizer.word_index) + 1
num_classes = len(y_train.unique())

print("\n" + "=" * 60)
print("LSTM DATA")
print("=" * 60)

print(f"\nVocabulary size: {vocab_size}")
print(f"Maximum sequence length: {max_length}")
print(f"Number of classes: {num_classes}")


# ============================================================
# 4. BUILD LSTM
# ============================================================

model = build_lstm_model(
    vocab_size=vocab_size,
    max_length=max_length,
    num_classes=num_classes
)


# ============================================================
# 5. DISPLAY MODEL
# ============================================================

print("\n" + "=" * 60)
print("LSTM MODEL")
print("=" * 60)

model.summary()


# ============================================================
# 6. TRAIN LSTM
# ============================================================

print("\n" + "=" * 60)
print("TRAINING LSTM")
print("=" * 60)

history = train_lstm_model(
    model=model,
    X_train=X_train,
    y_train=y_train,
    class_weights=class_weights,
    epochs=5,
    batch_size=32
)


print("\n" + "=" * 60)
print("LSTM TRAINING COMPLETED")
print("=" * 60)