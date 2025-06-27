# 👁️ Eye Gaze Estimation from Grayscale Eye Images  
**Predicting 3D Gaze Vectors using CNN, ViT, RNN & Random Forest**  
**Authors:** Jabed Miah, Gurpreet Singh, Prince Rana, Joceline A. Borja  
**Course:** Machine Learning for Neural and Biological Data – Spring 2025  

---

## 🔍 Project Overview
This project aims to estimate a person’s 3D gaze direction from eye images and head pose data using machine learning. We used the [4Quant Eye Gaze dataset](https://www.kaggle.com/datasets/4quant/eye-gaze) with over 50,000 annotated synthetic eye images to train and evaluate four models:

- ✅ **CNN** – Convolutional Neural Network (best performer)
- 📦 **ViT** – Vision Transformer (custom patch embedding)
- 🔁 **RNN** – Recurrent Neural Network (sequence modeling)
- 🌲 **Random Forest** – Baseline ensemble model

---

## 📁 Dataset
- Source: [Kaggle: 4Quant Eye Gaze Dataset](https://www.kaggle.com/datasets/4quant/eye-gaze)  
- Contains grayscale eye images with:
  - Gaze vectors (X, Y, Z)
  - Head pose (pitch, roll, yaw)
  - Iris size, pupil size, and metadata

Only synthetic images with labeled gaze vectors were used for model training.

---

## 🧪 Methodology
- Eye image preprocessing and normalization
- Gaze vector + head pose extraction
- Flattened image rows for RF and RNN; 2D inputs for CNN and ViT
- Evaluation Metrics:
  - ✅ Mean Squared Error (MSE)
  - ✅ R² Score
- Train/Test/Validation: 66/17/17 split

---

## 📊 Results

| Model          | MSE     | R² Score |
|----------------|---------|----------|
| CNN            | 0.001   | **0.9980** |
| ViT            | 0.004   | 0.9955   |
| RNN            | 0.008   | 0.9874   |
| Random Forest  | ~       | Moderate |

---

## 📈 Visualizations
- 3D scatter and quiver plots (predicted vs actual gaze vectors)
- Error histograms to detect outliers
- Focused analysis on top 1% largest errors

---

## 🧠 Lessons Learned
- CNNs are most effective for spatial image data
- RNNs underperform when spatial structure is flattened
- Random Forests work well with structured + tabular inputs
- Proper normalization and preprocessing drastically improve model quality

---

## 🔭 Future Work
- Use real eye images for realism
- Add data augmentation (brightness, flipping, jitter)
- Explore use in autism detection, AR/VR, eye-based control systems
- Improve baseline models using hyperparameter tuning and dimensionality reduction

---

## ⚙️ Technologies
- Python 3.9+
- TensorFlow, Keras
- Scikit-learn
- NumPy, Pandas, Matplotlib
- Google Colab / Jupyter

---

## 🚀 Getting Started


# Clone the repo
```bash
git clone https://github.com/yourusername/eye-gaze-estimation.git
cd eye-gaze-estimation
```
# (Optional) Create a virtual environment
```bash
python -m venv gaze-env
source gaze-env/bin/activate  # or use 'gaze-env\Scripts\activate' on Windows
```
# Install dependencies
```bash
pip install -r requirements.txt
```

## 👉File Path and Structure
📦 eye-gaze-estimation/
├── data/                # Contains downloaded dataset (optional)
├── models/              # Saved models (CNN, ViT, etc.)
├── notebooks/           # Jupyter/Colab notebooks for each model
├── src/                 # Python scripts for preprocessing and training
├── README.md
└── requirements.txt     # Required Python packages

**🙏 Acknowledgments**
4Quant Eye Gaze Dataset – Kaggle
