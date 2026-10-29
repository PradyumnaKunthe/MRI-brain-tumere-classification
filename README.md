
# MRI Brain Tumor Detection

An educational deep learning and machine learning project that classifies brain MRI images into four categories: glioma, meningioma, no tumor, and pituitary.

## Models

1. **Custom CNN** — three convolutional blocks with 32, 64, and 128 filters.
2. **MobileNetV2 + SVM** — pretrained MobileNetV2 feature extraction with an SVM classifier.
3. **MobileNetV2 + Random Forest** — MobileNetV2 features classified using Random Forest.
4. **MobileNetV2 + Logistic Regression** — MobileNetV2 features classified using Logistic Regression.
5. **Combined CNN + MobileNetV2 + SVM** — concatenated features from the custom CNN and MobileNetV2, classified using SVM.

## Experimental Results

Measured on a held-out test set of 2,414 MRI images.

| Model | Test Accuracy |
|---|---:|
| Custom CNN | 64.79% |
| MobileNetV2 + SVM | 92.50% |
| MobileNetV2 + Random Forest | 84.71% |
| MobileNetV2 + Logistic Regression | 89.06% |
| Combined CNN + MobileNetV2 + SVM | 92.42% |

MobileNetV2 + SVM achieved the highest test accuracy in this experiment. Feature fusion did not improve upon that result in this run.

## Technology Stack

- Python 3.11
- TensorFlow / Keras
- NumPy and Pandas
- OpenCV and Pillow
- Scikit-learn
- Matplotlib and Seaborn
- Joblib
- Streamlit

## Project Structure

```text
MRI-Brain-Tumor-Detection/
├── dataset/                    # Not included in Git
├── models/                     # Not included in Git
├── outputs/                    # Not included in Git
├── preprocessing.py
├── dataset_loader.py
├── train_cnn.py
├── train_hybrid.py
├── train_combined_hybrid.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd MRI-Brain-Tumor-Detection
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Prepare the dataset

Place your dataset under:

```text
dataset/
└── Epic and CSCR hospital Dataset/
    ├── Train/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── notumor/
    │   └── pituitary/
    └── Test/
        ├── glioma/
        ├── meningioma/
        ├── notumor/
        └── pituitary/
```

Run the preprocessing pipeline:

```powershell
python preprocessing.py
python dataset_loader.py
```

### 5. Train the models

```powershell
python train_cnn.py
python train_hybrid.py
python train_combined_hybrid.py
```

Training is not needed every time the application is launched, provided the compatible trained model files are available.

### 6. Launch the application

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## Data and Model Files

The MRI dataset, trained models, and generated outputs are excluded from this Git repository. Obtain the dataset separately and train the models, or arrange to copy the compatible saved model files into the `models/` folder before running the full application.

The first MobileNetV2 setup may require internet access to download pretrained ImageNet weights.

## Limitations

This project is an educational research prototype, not a clinically validated diagnostic system. Its predictions must not be used as a substitute for professional medical evaluation.
