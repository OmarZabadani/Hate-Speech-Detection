from Preprocessed_Data import prepare_dataset
from RNN import build_rnn_model, train_rnn_model


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
# 2. CHECK LABELS
# ============================================================

print("\n" + "=" * 60)
print("CHECKING DATA FOR RNN")
print("=" * 60)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")

print(f"\ny_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")

print(f"\ny_train type: {y_train.dtype}")
print(f"y_test type:  {y_test.dtype}")

print(f"\nUnique training labels: {sorted(y_train.unique())}")


# ============================================================
# 3. MODEL INFORMATION
# ============================================================

vocab_size = len(tokenizer.word_index) + 1
num_classes = len(y_train.unique())


print(f"\nVocabulary size: {vocab_size}")
print(f"Maximum sequence length: {max_length}")
print(f"Number of classes: {num_classes}")


# ============================================================
# 4. BUILD RNN
# ============================================================

model = build_rnn_model(
    vocab_size=vocab_size,
    max_length=max_length,
    num_classes=num_classes
)


# ============================================================
# 5. DISPLAY MODEL
# ============================================================

print("\n" + "=" * 60)
print("RNN MODEL")
print("=" * 60)

model.summary()


# ============================================================
# 6. TRAIN RNN
# ============================================================

print("\n" + "=" * 60)
print("TRAINING RNN")
print("=" * 60)

history = train_rnn_model(
    model=model,
    X_train=X_train,
    y_train=y_train,
    epochs=5,
    batch_size=32
)


print("\n" + "=" * 60)
print("RNN TRAINING COMPLETED")
print("=" * 60)