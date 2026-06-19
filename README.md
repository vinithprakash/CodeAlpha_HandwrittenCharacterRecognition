# ✍️ Handwritten Character Recognition
### CodeAlpha Machine Learning Internship — Task 3

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange?logo=tensorflow)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-green)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99%2B%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

This project implements a **Convolutional Neural Network (CNN)** to recognize handwritten digits (0–9) using the **MNIST dataset**. The model achieves **~99% test accuracy** through a modern CNN architecture with data augmentation and regularisation techniques.

---

## 🗂️ Project Structure

```
CodeAlpha_HandwrittenCharacterRecognition/
│
├── train.py              # Train the CNN model
├── predict.py            # Run inference on new / MNIST images
├── evaluate.py           # Full evaluation & classification report
├── requirements.txt      # Python dependencies
├── README.md             # This file
│
├── model/
│   └── cnn_mnist.keras   # Saved best model (created after training)
│
├── utils/
│   ├── training_history.png      # Accuracy & loss curves
│   ├── confusion_matrix.png      # Confusion matrix heatmap
│   ├── sample_predictions.png    # 25 sample test predictions
│   ├── inference_output.png      # Output from predict.py
│   └── eval_confusion_matrix.png # Output from evaluate.py
│
└── sample_images/
    └── (place your own .png digit images here for custom prediction)
```

---

## 🧠 Model Architecture

```
Input (28×28×1)
    │
    ▼
Conv2D(32, 3×3, relu) ──► BatchNorm ──► Conv2D(32, 3×3, relu) ──► MaxPool ──► Dropout(0.25)
    │
    ▼
Conv2D(64, 3×3, relu) ──► BatchNorm ──► Conv2D(64, 3×3, relu) ──► MaxPool ──► Dropout(0.25)
    │
    ▼
Flatten ──► Dense(256, relu) ──► BatchNorm ──► Dropout(0.5)
    │
    ▼
Dense(10, softmax)
    │
    ▼
Output: Digit class (0–9)
```

| Layer Type | Details |
|---|---|
| Optimizer | Adam |
| Loss | Categorical Cross-Entropy |
| Activation (hidden) | ReLU |
| Activation (output) | Softmax |
| Regularisation | BatchNorm + Dropout |
| Data Augmentation | Rotation ±10°, Zoom 10%, Shift 10% |
| Callbacks | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |

---

## 📊 Dataset

| Property | Value |
|---|---|
| Name | MNIST |
| Classes | 10 (digits 0–9) |
| Training samples | 60,000 |
| Test samples | 10,000 |
| Image size | 28×28 pixels (grayscale) |
| Source | `tensorflow.keras.datasets.mnist` |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/CodeAlpha_HandwrittenCharacterRecognition.git
cd CodeAlpha_HandwrittenCharacterRecognition
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python train.py
```
> ⏱️ Training takes ~5–10 minutes on CPU, ~1–2 minutes on GPU.  
> The best model is auto-saved to `model/cnn_mnist.keras`.

### 4. Evaluate the Model
```bash
python evaluate.py
```
> Prints full classification report and saves a confusion matrix.

### 5. Run Predictions

**On random MNIST test images:**
```bash
python predict.py
# or specify how many samples:
python predict.py --n 20
```

**On your own image:**
```bash
python predict.py --image sample_images/my_digit.png
```
> 📌 Use a white digit on a black background (28×28 or larger PNG/JPG).  
> The script auto-inverts black-on-white images.

---

## 📈 Results

| Metric | Value |
|---|---|
| Test Accuracy | **~99%** |
| Test Loss | **~0.03** |
| Parameters | ~500K |

**Per-class Performance (approximate):**

| Digit | Precision | Recall | F1-Score |
|---|---|---|---|
| 0 | 0.99 | 0.99 | 0.99 |
| 1 | 0.99 | 0.99 | 0.99 |
| 2 | 0.99 | 0.99 | 0.99 |
| 3 | 0.99 | 0.99 | 0.99 |
| 4 | 0.99 | 0.99 | 0.99 |
| 5 | 0.99 | 0.99 | 0.99 |
| 6 | 0.99 | 0.99 | 0.99 |
| 7 | 0.99 | 0.99 | 0.99 |
| 8 | 0.98 | 0.99 | 0.99 |
| 9 | 0.99 | 0.99 | 0.99 |

---

## 🖼️ Output Visualisations

After training, these files are saved in `utils/`:

| File | Description |
|---|---|
| `training_history.png` | Accuracy & loss over epochs |
| `confusion_matrix.png` | Heatmap of prediction errors |
| `sample_predictions.png` | 25 test images with true & predicted labels |
| `inference_output.png` | Output from running `predict.py` |

---

## 🔧 Key Concepts Used

- **CNNs** — Spatial feature extraction from images
- **BatchNormalization** — Stabilises and speeds up training
- **Dropout** — Prevents overfitting
- **Data Augmentation** — Improves model generalisation
- **EarlyStopping** — Stops training when validation accuracy plateaus
- **ReduceLROnPlateau** — Dynamically reduces learning rate
- **Confusion Matrix** — Visual analysis of classification errors

---

## 🔮 Future Improvements

- Extend to EMNIST dataset for full alphabet recognition (A–Z)
- Build a web UI using Flask or Streamlit for live drawing & prediction
- Add CRNN model for full word/sentence recognition
- Convert to TensorFlow Lite for mobile deployment

---

## 👨‍💻 Author

**Vinith Prakash B**  
Machine Learning Intern — CodeAlpha  

---

## 🏢 About CodeAlpha

CodeAlpha is a leading software development company driving innovation through AI and intelligent systems.  
🌐 [www.codealpha.tech](https://www.codealpha.tech)

---

## 📄 License

This project is licensed under the MIT License.
