"""
CodeAlpha Internship - Task 3: Handwritten Character Recognition
predict.py — Run inference on a single image or random MNIST samples
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

MODEL_PATH = "model/cnn_mnist.keras"
IMG_SIZE   = 28


def predict_from_mnist(n: int = 10):
    """Pick n random MNIST test images and predict them."""
    model = load_model(MODEL_PATH)
    (_, _), (X_test, y_test) = mnist.load_data()

    idx = np.random.choice(len(X_test), n, replace=False)
    images = X_test[idx].reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0
    preds  = np.argmax(model.predict(images, verbose=0), axis=1)

    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.array(axes).flat

    for ax, img, true, pred in zip(axes, X_test[idx], y_test[idx], preds):
        ax.imshow(img, cmap="gray")
        color = "green" if true == pred else "red"
        ax.set_title(f"True: {true}\nPred: {pred}", color=color, fontsize=10)
        ax.axis("off")

    for ax in list(axes)[n:]:
        ax.axis("off")

    plt.suptitle("MNIST Predictions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("utils/inference_output.png", dpi=150)
    plt.show()
    print(f"\nAccuracy on sample: {(y_test[idx] == preds).mean() * 100:.1f}%")
    print("Saved: utils/inference_output.png")


def predict_custom_image(path: str):
    """Predict digit from a custom grayscale image file."""
    try:
        from PIL import Image
    except ImportError:
        print("Install Pillow:  pip install pillow")
        sys.exit(1)

    model = load_model(MODEL_PATH)

    img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32")

    # MNIST is white-on-black; invert if the image is black-on-white
    if arr.mean() > 127:
        arr = 255 - arr

    arr = arr / 255.0
    arr = arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    probs = model.predict(arr, verbose=0)[0]
    pred  = np.argmax(probs)

    print(f"\nPredicted digit : {pred}  (confidence: {probs[pred] * 100:.1f}%)")

    plt.figure(figsize=(4, 4))
    plt.imshow(arr.reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
    plt.title(f"Prediction: {pred}  ({probs[pred]*100:.1f}%)", fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("utils/custom_prediction.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handwritten digit predictor")
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a custom image. Omit to use random MNIST samples."
    )
    parser.add_argument(
        "--n", type=int, default=10,
        help="Number of random MNIST samples to predict (default 10)."
    )
    args = parser.parse_args()

    if args.image:
        predict_custom_image(args.image)
    else:
        predict_from_mnist(args.n)
