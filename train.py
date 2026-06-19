"""
CodeAlpha Internship - Task 3: Handwritten Character Recognition
Author: [Your Name]
Description: CNN-based model to recognize handwritten digits using MNIST dataset
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix

# ─── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE    = 28
NUM_CLASSES = 10
EPOCHS      = 20
BATCH_SIZE  = 128
MODEL_PATH  = "model/cnn_mnist.keras"
# ──────────────────────────────────────────────────────────────────────────────


def load_and_preprocess():
    """Load MNIST and normalise to [0,1]."""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0
    X_test  = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0

    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_test_cat  = to_categorical(y_test,  NUM_CLASSES)

    print(f"Train samples : {X_train.shape[0]}")
    print(f"Test  samples : {X_test.shape[0]}")
    return X_train, y_train, y_train_cat, X_test, y_test, y_test_cat


def build_model():
    """Build a compact but effective CNN."""
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation="relu", padding="same",
               input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Classifier
        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


def get_callbacks():
    os.makedirs("model", exist_ok=True)
    return [
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]


def plot_history(history):
    os.makedirs("utils", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"],     label="Train Acc",  color="#4C72B0")
    axes[0].plot(history.history["val_accuracy"], label="Val   Acc",  color="#DD8452")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train Loss", color="#4C72B0")
    axes[1].plot(history.history["val_loss"], label="Val   Loss", color="#DD8452")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("utils/training_history.png", dpi=150)
    plt.close()
    print("Saved: utils/training_history.png")


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(NUM_CLASSES),
                yticklabels=range(NUM_CLASSES))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("utils/confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved: utils/confusion_matrix.png")


def plot_samples(X_test, y_test, y_pred):
    """Show 25 random test samples with true vs predicted labels."""
    idx = np.random.choice(len(X_test), 25, replace=False)
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for ax, i in zip(axes.flat, idx):
        ax.imshow(X_test[i].reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
        color = "green" if y_test[i] == y_pred[i] else "red"
        ax.set_title(f"T:{y_test[i]} P:{y_pred[i]}", color=color, fontsize=9)
        ax.axis("off")
    plt.suptitle("Sample Predictions (green=correct, red=wrong)", fontsize=13)
    plt.tight_layout()
    plt.savefig("utils/sample_predictions.png", dpi=150)
    plt.close()
    print("Saved: utils/sample_predictions.png")


def main():
    print("=" * 55)
    print("  CodeAlpha — Handwritten Character Recognition")
    print("=" * 55)

    X_train, y_train, y_train_cat, X_test, y_test, y_test_cat = load_and_preprocess()

    # Data augmentation (light, so we don't distort digits too much)
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.10,
        width_shift_range=0.10,
        height_shift_range=0.10,
    )
    datagen.fit(X_train)

    model = build_model()

    print("\nTraining …\n")
    history = model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_test, y_test_cat),
        callbacks=get_callbacks(),
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTest Accuracy : {acc * 100:.2f}%")
    print(f"Test Loss     : {loss:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_history(history)
    plot_confusion(y_test, y_pred)
    plot_samples(X_test, y_test, y_pred)

    print("\nAll done! Model saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()
