"""
CodeAlpha Internship - Task 3: Handwritten Character Recognition
evaluate.py — Full evaluation report on the MNIST test set
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH  = "model/cnn_mnist.keras"
NUM_CLASSES = 10
IMG_SIZE    = 28


def main():
    print("Loading model …")
    model = load_model(MODEL_PATH)

    (_, _), (X_test, y_test) = mnist.load_data()
    X_test_n = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0
    y_test_cat = to_categorical(y_test, NUM_CLASSES)

    loss, acc = model.evaluate(X_test_n, y_test_cat, verbose=1)
    print(f"\n{'='*45}")
    print(f"  Test Accuracy : {acc * 100:.2f}%")
    print(f"  Test Loss     : {loss:.4f}")
    print(f"{'='*45}\n")

    y_pred = np.argmax(model.predict(X_test_n, verbose=0), axis=1)

    print("Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=[f"Digit {i}" for i in range(NUM_CLASSES)]))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr",
                xticklabels=range(NUM_CLASSES),
                yticklabels=range(NUM_CLASSES))
    plt.title("Confusion Matrix — MNIST Test Set", fontsize=14)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("utils/eval_confusion_matrix.png", dpi=150)
    plt.close()
    print("\nSaved: utils/eval_confusion_matrix.png")

    # Most-confused pairs
    np.fill_diagonal(cm, 0)
    top5 = np.argsort(cm.flatten())[-5:][::-1]
    print("\nTop-5 most confused pairs (true → predicted):")
    for flat_idx in top5:
        r, c = divmod(flat_idx, NUM_CLASSES)
        print(f"  {r} → {c}  ({cm[r, c]} errors)")


if __name__ == "__main__":
    main()
