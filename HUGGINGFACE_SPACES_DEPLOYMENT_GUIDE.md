# 🚀 Hugging Face Spaces Docker Deployment Guide

## MRI Brain Tumor Detection Platform (React + FastAPI + TensorFlow)

This comprehensive guide walks you through deploying the unified **MRI Brain Tumor Detection Platform** to **Hugging Face Spaces** using the **Docker SDK**.

---

## 📋 Architecture & Resource Summary

| Component | Specification | Deployment Port |
| :--- | :--- | :---: |
| **Frontend UI** | React 18, Vite 5, Tailwind CSS, Recharts | Built into `frontend/dist` |
| **Backend API** | FastAPI, Uvicorn, Python-Multipart | Port `7860` |
| **Unified Runtime** | Multi-stage Docker container | Port `7860` (Single Port) |
| **Trained Models** | Custom CNN, MobileNetV2 + SVM / RF / LR / Combined | Preloaded in `models/` |
| **Hugging Face Tier** | Free CPU Tier (2 vCPU, 16 GB RAM, 50 GB Disk) | Zero-Cost Deployment |

---

## 🛠️ Prerequisites

1. A **Hugging Face** account: [Sign up here](https://huggingface.co/join) if you do not have one.
2. **Git** installed on your machine (`git --version`).
3. **Git LFS** (Large File Storage) installed:
   ```bash
   git lfs --version
   ```
   *(If not installed: Windows: `winget install GitHub.GitLFS` or download from [git-lfs.com](https://git-lfs.com/))*

---

## 🚀 Step-by-Step Deployment Instructions

### Step 1: Create a New Space on Hugging Face

1. Navigate to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Fill in the following settings:
   - **Space name:** e.g., `mri-brain-tumor-detection`
   - **License:** `mit` (or your preferred license)
   - **Select the Space SDK:** Choose **Docker** 🐳
   - **Docker template:** Choose **Blank**
   - **Space Hardware:** Select **CPU basic (Free • 2 vCPU • 16 GB RAM)**
   - **Space visibility:** Public (or Private)
3. Click **Create Space**.

---

### Step 2: Configure Git LFS for Trained Model Artifacts

Hugging Face Spaces requires all files larger than 10 MB (such as `.joblib` and `.keras` model weights) to be tracked with Git LFS. The repository already includes a configured `.gitattributes` file.

In your local repository root, initialize Git LFS:
```bash
git lfs install
```

Ensure tracking is registered:
```bash
git lfs track "models/*.joblib" "models/*.keras"
```

---

### Step 3: Link Your Local Repository to Hugging Face Spaces

Add the Hugging Face Space as a new Git remote named `space`:

```bash
# Replace YOUR_USERNAME and YOUR_SPACE_NAME with your actual Hugging Face details:
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
```

> **Note on Authentication:** When Git prompts for a password, generate and use an **Access Token** with `write` permissions from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Do not use your regular account password.

---

### Step 4: Include Model Artifacts for Hugging Face Deployment

The local `.gitignore` prevents large model files from polluting lightweight GitHub repositories. For your Hugging Face Space, the trained model files **must** be pushed so the application can run inference.

1. Temporarily allow the model files and sample dataset to be added:
   ```bash
   git add -f models/
   git add -f outputs/
   git add -f "dataset/Epic and CSCR hospital Dataset/Test"
   git add Dockerfile .dockerignore .gitattributes README.md
   ```

2. Commit the deployment bundle:
   ```bash
   git commit -m "chore: package trained models and docker configuration for HF Spaces"
   ```

3. Push directly to your Hugging Face Space:
   ```bash
   git push space main
   ```

---

### Step 5: Monitor the Container Build & Verification

1. Go to your Space page on Hugging Face (`https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`).
2. Click the **Building** badge at the top to watch live container logs:
   - **Stage 1 (Node.js):** Builds the React SPA via Vite into `/app/frontend/dist`.
   - **Stage 2 (Python 3.11):** Installs TensorFlow, scikit-learn, and FastAPI requirements, then copies preloaded models.
   - **Container Start:** Uvicorn starts on `0.0.0.0:7860`.
3. Once the build finishes, status switches to **Running** (green).
4. Your application is now live at:
   `https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space`

---

## 🧪 Testing the Docker Container Locally (Optional)

Before pushing to the cloud, you can optionally test the exact container image on your local machine using Docker Desktop:

### 1. Build the Docker Image:
```bash
docker build -t mri-tumor-detection .
```

### 2. Run the Container:
```bash
docker run -p 7860:7860 --name mri-app mri-tumor-detection
```

### 3. Open in Browser:
- **Application UI:** [http://localhost:7860](http://localhost:7860)
- **Interactive API Docs:** [http://localhost:7860/docs](http://localhost:7860/docs)
- **Container Health Check:** [http://localhost:7860/api/health](http://localhost:7860/api/health)

### 4. Stop Container:
```bash
docker stop mri-app && docker rm mri-app
```

---

## 🔍 Troubleshooting & Common Questions

### 1. Why does Hugging Face Spaces require port 7860?
Hugging Face Spaces routes HTTP traffic exclusively to internal container port `7860`. The `Dockerfile` and `integration_api/main.py` explicitly bind to `0.0.0.0:${PORT:-7860}`, ensuring immediate compatibility.

### 2. Why does the container use a non-root user (UID 1000)?
For security reasons, Hugging Face Spaces runs Docker containers with an unprivileged user having user ID `1000`. The `Dockerfile` creates `user` (`RUN useradd -m -u 1000 user`) and sets `USER user` to prevent permission denied errors on temporary files or model caches.

### 3. Are the machine learning models retrained inside Docker?
**No.** All 6 trained model checkpoints (`models/*.keras`, `models/*.joblib`) are pre-trained and bundled into the container. Cold-start inference loads models in ~2 seconds upon startup.

### 4. Will the React SPA client-side routes (e.g. `/analysis`, `/comparison`) work without 404s?
**Yes.** `integration_api/main.py` includes a FastAPI fallback route (`@app.get("/{full_path:path}")`) that routes all non-API paths back to `frontend/dist/index.html`. Refreshing pages or navigating directly will never yield a 404.
