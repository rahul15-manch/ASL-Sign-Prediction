# 🖐️ ASL Sign Language Recognition

## 💡 Overview

A deep learning-based project to recognize American Sign Language (ASL) letters from images.
Built with TensorFlow, OpenCV, and Streamlit — upload a hand sign image and get a real-time prediction with optional text-to-speech output.

---

## 🚀 Features

- 📸 **Image Upload** – Upload single or multiple hand sign images for prediction
- 📷 **Webcam Support** – Capture a live photo directly in the app
- 🧠 **Deep Learning Model** – Improved CNN with BatchNormalization and Dropout
- ⚡ **Preprocessing** – Images resized, normalized, and augmented for generalization
- 🌐 **Streamlit App** – Interactive web UI for easy deployment
- 🔊 **Text-to-Speech** – Converts predicted output to speech via gTTS / pyttsx3

---

## 📂 Project Structure

```
ASL-Sign-Prediction/
│── app.py                     # Streamlit web app
│── requirements.txt           # Python dependencies
│── README.md                  # Project documentation
│── src/
│   ├── train.py               # Training pipeline (with EarlyStopping, datagen fix)
│   ├── models.py              # CNN model architecture (BatchNorm + Dropout)
│   ├── test.py                # Model evaluation script
│   ├── preprocessing.py       # Image preprocessing utilities
│   ├── inference.py           # Single-image inference
│   ├── helper.py              # Plotting, model load/save utilities
│   └── tts.py                 # Text-to-speech integration
│── models/
│   ├── sign_model.h5          # Trained CNN model (best checkpoint)
│   ├── sign_model_backup.h5   # Backup of previous model
│   ├── class_indices.json     # Class label mapping (A–Z + space, delete, nothing)
│   └── training_history.png   # Accuracy & loss curves
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/rahul15-manch/ASL-Sign-Prediction.git
cd ASL-Sign-Prediction

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## 📊 Dataset

Uses the [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) by **grassknoted** on Kaggle.

| Split | Images | Classes |
|-------|--------|---------|
| Train | 87,000 | 29 (A–Z + space, delete, nothing) |
| Test  | 29     | 29 (1 per class) |

**Auto-download via kagglehub:**

```python
import kagglehub
path = kagglehub.dataset_download("grassknoted/asl-alphabet")
print("Dataset path:", path)
```

> The training script (`src/train.py`) automatically resolves the dataset path using `kagglehub`.

---

## 🧠 Model Architecture

Custom CNN with regularization to reduce overfitting:

```
Input (64×64×3)
  → Conv2D(32) → BatchNorm → MaxPool
  → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
  → Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
  → Flatten → Dense(256) → Dropout(0.5)
  → Softmax(29)
```

| Parameter    | Value                   |
|--------------|------------------------|
| Input size   | 64×64 RGB              |
| Optimizer    | Adam                   |
| Loss         | Categorical Crossentropy |
| Best Epoch   | 8 (via EarlyStopping)  |
| Val Accuracy | ~79%                   |
| **Test Accuracy** | **~93%** (A–Z letters) |

---

## 🏋️ Training

### Data Augmentation (Train only)
- Rotation ±15°, Zoom 20%, Shear 20%, Horizontal Flip
- Validation data uses **rescaling only** (no augmentation)

### Callbacks
- `EarlyStopping` — patience=5, restores best weights
- `ReduceLROnPlateau` — patience=3, factor=0.5

### Retrain from scratch:
```bash
source venv/bin/activate
python src/train.py
```

Training history is saved to `models/training_history.png`.

---

## 📈 Training History

![Training History](https://raw.githubusercontent.com/rahul15-manch/ASL-Sign-Prediction/main/models/training_history.png)

---

## 🌐 Deployment on Streamlit Cloud

1. Push code to GitHub
2. Ensure `requirements.txt` includes all dependencies
3. Add `runtime.txt` with `python-3.10.13` if needed
4. Deploy on [Streamlit Cloud](https://streamlit.io/cloud)

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow/Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-ffffff?style=for-the-badge&logo=plotly&logoColor=black)
![Seaborn](https://img.shields.io/badge/Seaborn-0099CC?style=for-the-badge&logoColor=white)
![kagglehub](https://img.shields.io/badge/kagglehub-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![gTTS](https://img.shields.io/badge/gTTS-FFDD00?style=for-the-badge&logo=google&logoColor=black)

---

## 📱 Application UI

![Demonstration](https://raw.githubusercontent.com/rahul15-manch/ASL-Sign-Prediction/main/0011.png)

## 🟢 Output

![Output](https://raw.githubusercontent.com/rahul15-manch/ASL-Sign-Prediction/refs/heads/main/0012.png)

---

## 👨‍💻 Authors

### 🌐 Connect with us

| Name | LinkedIn |
|------|----------|
| **Rahul Manchanda** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rahul-manchanda-3959b120a/) |
| **Tanishka** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/tanishka-mukhi09/) |
| **Kashish** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kashish-rana-6116691b5/) |
