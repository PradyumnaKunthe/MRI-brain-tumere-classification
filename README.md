---
title: MRI Brain Tumor Detection
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Multi-model MRI brain tumor detection with MobileNetV2 and SVM
---

# 🧠 MRI Brain Tumor Detection Using Deep Learning & Machine Learning

An academic and clinical research application that categorizes brain Magnetic Resonance Imaging (MRI) scans into four distinct diagnostic classes:
1. **Glioma**
2. **Meningioma**
3. **No Tumor (Healthy Control)**
4. **Pituitary Tumor**

---

## 🔬 Model Architectures

The platform compares five distinct modeling strategies across standalone Deep Learning, Classical Machine Learning, and Multi-Stream Feature Fusion:

1. **MobileNetV2 + SVM (Best Classifier)**: Pretrained ImageNet weights with Global Average Pooling (1,280-d feature vector) classified using a 3-fold Stratified GridSearchCV-tuned Support Vector Machine with an RBF kernel.
2. **Combined CNN + MobileNetV2 + SVM**: Concatenates a 128-dimensional dense embedding from a Custom CNN with the 1,280-dimensional MobileNetV2 vector (total 1,408-d feature space), classified by an SVM.
3. **MobileNetV2 + Logistic Regression**: Fast linear convex baseline on 1,280-d features.
4. **MobileNetV2 + Random Forest**: Non-linear ensemble decision tree baseline (150 estimators, balanced subsampling).
5. **Custom CNN**: Standalone 3-block convolutional neural network (32, 64, 128 filters) trained from scratch.

---

## 📊 Empirical Evaluation Results

All models were evaluated on an identical held-out test split of **2,414 clinical MRI images** never seen during training or cross-validation:

| Rank | Model Architecture | Validation Accuracy | Test Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **MobileNetV2 + SVM (Tuned)** | **96.89%** | **92.50%** | **92.48%** | **92.50%** | **92.42%** |
| 🥈 | **Combined CNN + MobileNetV2 + SVM** | **97.36%** | **92.42%** | **92.44%** | **92.42%** | **92.34%** |
| 🥉 | **MobileNetV2 + Logistic Regression** | 93.63% | **89.06%** | 89.06% | 89.06% | 88.93% |
| 4 | **MobileNetV2 + Random Forest** | 90.88% | **84.71%** | 84.21% | 84.71% | 84.15% |
| 5 | **Custom Standalone CNN** | 68.14% | **64.79%** | 64.92% | 64.79% | 64.21% |

---

## 📁 Dataset Statistics (12,064 Curated Images)

The dataset was partitioned using an 80/20 train-validation split and an independent 20% held-out test set:

| Pathology Category | Training Images | Validation Images | Held-Out Test Images | Total Images |
| :--- | :---: | :---: | :---: | :---: |
| **Glioma** | 2,414 | 604 | 755 | **3,773** |
| **Meningioma** | 1,747 | 436 | 546 | **2,729** |
| **No Tumor** | 1,556 | 389 | 487 | **2,432** |
| **Pituitary** | 2,003 | 501 | 626 | **3,130** |
| **Total Corpus** | **7,720** | **1,930** | **2,414** | **12,064** |

---

## 🛠️ Technology Stack

- **Frontend:** React 18, Vite 5, Tailwind CSS, Lucide React, Recharts, Axios
- **Backend API:** FastAPI, Uvicorn, Python-Multipart
- **Machine Learning:** TensorFlow / Keras 2.21, Scikit-learn 1.9, Joblib 1.6
- **Scientific Computing:** NumPy, Pandas, OpenCV, Pillow, Matplotlib, Seaborn
- **Fallback UI:** Streamlit 1.66
- **Containerization:** Docker SDK (Hugging Face Spaces)

---

## 🚀 Running Locally

### 1. Clone the repository and navigate to the directory:
```bash
git clone https://github.com/PradyumnaKunthe/MRI-brain-tumere-classification.git
cd MRI-Brain-Tumor-Detection
```

### 2. Create and activate virtual environment:
```powershell
# Windows
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Run the Applications:

#### Option A: Unified React.js + FastAPI Platform (Recommended)
FastAPI serves the compiled React production frontend and all API endpoints together on a single port:
```bash
uvicorn integration_api.main:app --host 0.0.0.0 --port 8000
```
- Web Application: [http://localhost:8000](http://localhost:8000)
- Interactive API Docs (Swagger): [http://localhost:8000/docs](http://localhost:8000/docs)

#### Option B: React Vite Development Server (Hot Reload)
```bash
# Terminal 1: Backend
uvicorn integration_api.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend
npm install
npm run dev
```
- Dev Server: [http://localhost:5173](http://localhost:5173) (Proxies `/api` to port 8000)

#### Option C: Preserved Streamlit Application (Fallback)
```bash
python -m streamlit run app.py
```
- Streamlit Fallback: [http://localhost:8501](http://localhost:8501)

### 5. Run the QA Test Suite:
```bash
python run_tests.py
```

---

## 🐳 Deployment to Hugging Face Spaces (Docker SDK)

1. Log in to [Hugging Face](https://huggingface.co/) and click **New Space**.
2. Name your space (e.g. `mri-brain-tumor-detection`) and select **Docker** as the Space SDK.
3. Clone your new Hugging Face Space repository:
   ```bash
   git clone https://huggingface.co/spaces/<YOUR_USERNAME>/<SPACE_NAME>
   cd <SPACE_NAME>
   ```
4. Copy the project files (`app.py`, `models/`, `outputs/`, `Dockerfile`, `.dockerignore`, `.streamlit/`, `requirements.txt`, `README.md`) into your space directory.
5. Push using Git LFS for large model files (`*.joblib`, `*.keras`):
   ```bash
   git lfs install
   git lfs track "*.joblib" "*.keras"
   git add .gitattributes
   git add .
   git commit -m "Deploy MRI Brain Tumor Detection Suite"
   git push origin main
   ```
6. Hugging Face Spaces will automatically build the Docker container and serve the application on port `7860`.

---

## ⚠️ Medical Disclaimer

This application is an educational prototype and research demonstrator. It has not been approved as a diagnostic tool by regulatory agencies (FDA/CE) and must not be used as an independent basis for clinical decisions.
