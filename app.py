"""
MRI Brain Tumor Detection — Clinical AI Research & Diagnostic Suite.
Multi-model classification platform across Glioma, Meningioma, No Tumor, and Pituitary pathologies.
Integrates Custom CNN, Pretrained MobileNetV2 feature extraction with SVM, Random Forest,
Logistic Regression, and Fused Hybrid architecture.
"""

from pathlib import Path
import json
import time

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. DIRECTORIES & CONSTANTS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
DATASET_DIR = BASE_DIR / "dataset" / "Epic and CSCR hospital Dataset"
TEST_DATASET_DIR = DATASET_DIR / "Test"

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

CLASS_DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary",
}

CLASS_COLORS = {
    "glioma": "#D97706",        # Amber / Orange
    "meningioma": "#7C3AED",    # Purple
    "notumor": "#059669",       # Green
    "pituitary": "#0284C7",     # Cyan / Blue
}

CLASS_BG_COLORS = {
    "glioma": "#FFFBEB",
    "meningioma": "#F5F3FF",
    "notumor": "#ECFDF5",
    "pituitary": "#F0F9FF",
}

CLASS_ICONS = {
    "glioma": "⚠️",
    "meningioma": "🟣",
    "notumor": "✅",
    "pituitary": "🔵",
}

CNN_PATH = MODEL_DIR / "custom_cnn.keras"
MOBILENET_PATH = MODEL_DIR / "mobilenetv2_feature_extractor.keras"
COMBINED_SVM_PATH = MODEL_DIR / "combined_cnn_mobilenet_svm.joblib"

HYBRID_PATHS = {
    "MobileNetV2 + SVM":
        MODEL_DIR / "mobilenetv2_plus_svm.joblib",
    "MobileNetV2 + Random Forest":
        MODEL_DIR / "mobilenetv2_plus_random_forest.joblib",
    "MobileNetV2 + Logistic Regression":
        MODEL_DIR / "mobilenetv2_plus_logistic_regression.joblib",
}

# --------------------------------------------------
# 2. PAGE CONFIGURATION & RESPONSIVE STYLING
# --------------------------------------------------

st.set_page_config(
    page_title="MRI Brain Tumor Detection — Research Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Professional Medical AI Dashboard Styling */
    :root {
        --primary: #0284C7;
        --primary-dark: #0369A1;
        --border-color: rgba(148, 163, 184, 0.25);
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }
    
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 18px;
    }
    
    .metric-card {
        background: #FFFFFF;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    
    .badge-pill {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 2px 4px 2px 0;
    }

    .consensus-banner {
        border-radius: 12px;
        padding: 18px 22px;
        margin: 16px 0;
        border-left: 6px solid #0284C7;
        background: #F0F9FF;
    }
    
    .disclaimer-box {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 5px solid #EF4444;
        border-radius: 8px;
        padding: 14px 18px;
        color: #991B1B;
        font-size: 0.9rem;
        margin-top: 20px;
    }
    
    /* Touch Accessibility & Responsive Mobile Design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 1.6rem !important;
        }
        .sub-title {
            font-size: 0.9rem !important;
        }
        .stButton>button {
            width: 100% !important;
            min-height: 48px !important;
            font-size: 1rem !important;
        }
        .metric-card {
            padding: 12px !important;
        }
        img {
            max-width: 100% !important;
            height: auto !important;
        }
        div[data-testid="column"] {
            min-width: 100% !important;
            margin-bottom: 14px;
        }
    }
    button, input, select {
        min-height: 42px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 3. MODEL LOADING & CACHING
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_all_models():
    """Load all model artifacts with fault-tolerant status tracking."""
    status = {}
    models = {
        "cnn": None,
        "cnn_feature_model": None,
        "mobilenet": None,
        "combined_svm": None,
        "hybrid_models": {},
    }

    # 1. Custom CNN
    if CNN_PATH.is_file():
        try:
            cnn = tf.keras.models.load_model(CNN_PATH, compile=False)
            cnn_feature_model = tf.keras.Model(
                inputs=cnn.input,
                outputs=cnn.get_layer("dense").output,
                name="custom_cnn_feature_extractor",
            )
            models["cnn"] = cnn
            models["cnn_feature_model"] = cnn_feature_model
            status["Custom CNN"] = True
        except Exception as err:
            status["Custom CNN"] = f"Error: {err}"
    else:
        status["Custom CNN"] = "Model artifact missing"

    # 2. MobileNetV2 Feature Extractor
    if MOBILENET_PATH.is_file():
        try:
            mobilenet = tf.keras.models.load_model(
                MOBILENET_PATH,
                compile=False,
            )
            models["mobilenet"] = mobilenet
            status["MobileNetV2 Feature Extractor"] = True
        except Exception as err:
            status["MobileNetV2 Feature Extractor"] = f"Error: {err}"
    else:
        status["MobileNetV2 Feature Extractor"] = "Model artifact missing"

    # 3. Combined SVM
    if COMBINED_SVM_PATH.is_file():
        try:
            models["combined_svm"] = joblib.load(COMBINED_SVM_PATH)
            status["Combined CNN + MobileNetV2 + SVM"] = True
        except Exception as err:
            status["Combined CNN + MobileNetV2 + SVM"] = f"Error: {err}"
    else:
        status["Combined CNN + MobileNetV2 + SVM"] = "Model artifact missing"

    # 4. Hybrid Classifiers
    for name, path in HYBRID_PATHS.items():
        if path.is_file():
            try:
                models["hybrid_models"][name] = joblib.load(path)
                status[name] = True
            except Exception as err:
                status[name] = f"Error: {err}"
        else:
            status[name] = "Model artifact missing"

    return models, status


# Backward-compatible alias
load_models = load_all_models


# --------------------------------------------------
# 4. PREPROCESSING & RADIOLOGICAL TOOLS
# --------------------------------------------------

def prepare_image(pil_image: Image.Image):
    """
    Standardize image to 224x224 RGB and produce model inputs:
    - Custom CNN: range [0, 1]
    - MobileNetV2: range [-1, 1]
    """
    image = ImageOps.exif_transpose(pil_image).convert("RGB")
    image = image.resize((224, 224), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32)

    cnn_input = np.expand_dims(pixels / 255.0, axis=0)
    mobilenet_input = np.expand_dims(pixels / 127.5 - 1.0, axis=0)

    return image, cnn_input, mobilenet_input


def apply_clahe_filter(pil_image: Image.Image) -> Image.Image:
    """Enhance soft tissue contrast via CLAHE."""
    img_gray = np.array(pil_image.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)
    return Image.fromarray(enhanced)


def apply_edge_filter(pil_image: Image.Image) -> Image.Image:
    """Highlight tumor boundaries via Canny contour detection."""
    img_gray = np.array(pil_image.convert("L"))
    blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 60, 160)
    return Image.fromarray(edges)


def apply_inverted_filter(pil_image: Image.Image) -> Image.Image:
    """High-contrast inverted radiology lightbox view."""
    return ImageOps.invert(pil_image.convert("L"))


def class_name(label: int) -> str:
    """Convert integer label index to class name string."""
    idx = int(label)
    if not 0 <= idx < len(CLASS_NAMES):
        raise ValueError(f"Invalid label index: {idx}")
    return CLASS_NAMES[idx]


# --------------------------------------------------
# 5. INFERENCE HELPERS
# --------------------------------------------------

def predict_single_model(model, features):
    """Predict label and return full probability distribution where available."""
    predicted_label = int(model.predict(features)[0])
    predicted_class = class_name(predicted_label)
    prob_dist = {}
    score = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        for label, prob in zip(model.classes_, probabilities):
            prob_dist[class_name(label)] = float(prob)
        score = prob_dist.get(predicted_class)
    elif hasattr(model, "decision_function"):
        try:
            decision = model.decision_function(features)[0]
            exp_d = np.exp(decision - np.max(decision))
            softmax_p = exp_d / np.sum(exp_d)
            for idx, prob in enumerate(softmax_p):
                prob_dist[class_name(idx)] = float(prob)
            score = prob_dist.get(predicted_class)
        except Exception:
            score = None

    return predicted_class, score, prob_dist


def run_all_predictions(models_dict, cnn_input, mobilenet_input):
    """Execute all active models and calculate consensus agreement."""
    results = []

    # 1. Custom CNN
    if models_dict["cnn"] is not None:
        t0 = time.perf_counter()
        cnn_probs = models_dict["cnn"].predict(cnn_input, verbose=0)[0]
        cnn_time_ms = (time.perf_counter() - t0) * 1000

        cnn_idx = int(np.argmax(cnn_probs))
        cnn_pred_class = class_name(cnn_idx)
        cnn_prob_dist = {
            class_name(i): float(cnn_probs[i])
            for i in range(len(CLASS_NAMES))
        }
        results.append({
            "model": "Custom CNN",
            "type": "Standalone 3-Block CNN",
            "benchmark_accuracy": "64.79%",
            "predicted_class": cnn_pred_class,
            "confidence": float(cnn_probs[cnn_idx]),
            "probabilities": cnn_prob_dist,
            "inference_time_ms": cnn_time_ms,
        })

    # MobileNetV2 Feature Extraction
    mobilenet_feat = None
    cnn_feat = None

    if models_dict["mobilenet"] is not None:
        t0 = time.perf_counter()
        mobilenet_feat = models_dict["mobilenet"].predict(
            mobilenet_input,
            verbose=0,
        )
        mob_extract_time_ms = (time.perf_counter() - t0) * 1000
    else:
        mob_extract_time_ms = 0.0

    if models_dict["cnn_feature_model"] is not None:
        cnn_feat = models_dict["cnn_feature_model"].predict(
            cnn_input,
            verbose=0,
        )

    # 2. Combined Hybrid Model
    if (
        models_dict["combined_svm"] is not None
        and cnn_feat is not None
        and mobilenet_feat is not None
    ):
        combined_feat = np.concatenate([cnn_feat, mobilenet_feat], axis=1).astype(np.float32)
        t0 = time.perf_counter()
        pred_class, score, prob_dist = predict_single_model(
            models_dict["combined_svm"], combined_feat
        )
        comb_time_ms = (time.perf_counter() - t0) * 1000
        results.append({
            "model": "Combined CNN + MobileNetV2 + SVM",
            "type": "Fused Embedding (1408-d)",
            "benchmark_accuracy": "92.42%",
            "predicted_class": pred_class,
            "confidence": score,
            "probabilities": prob_dist,
            "inference_time_ms": comb_time_ms,
        })

    # 3-5. Hybrid Classifiers
    if mobilenet_feat is not None:
        benchmarks = {
            "MobileNetV2 + SVM": ("ImageNet Features + Tuned SVM", "92.50%"),
            "MobileNetV2 + Random Forest": ("ImageNet Features + Random Forest", "84.71%"),
            "MobileNetV2 + Logistic Regression": ("ImageNet Features + Logistic Regression", "89.06%"),
        }

        for name, clf in models_dict["hybrid_models"].items():
            if clf is not None:
                m_type, bench_acc = benchmarks.get(name, ("Hybrid", "N/A"))
                t0 = time.perf_counter()
                pred_class, score, prob_dist = predict_single_model(clf, mobilenet_feat)
                clf_time_ms = (time.perf_counter() - t0) * 1000 + mob_extract_time_ms
                results.append({
                    "model": name,
                    "type": m_type,
                    "benchmark_accuracy": bench_acc,
                    "predicted_class": pred_class,
                    "confidence": score,
                    "probabilities": prob_dist,
                    "inference_time_ms": clf_time_ms,
                })

    # Consensus calculation
    if results:
        preds = [r["predicted_class"] for r in results]
        class_counts = pd.Series(preds).value_counts()
        consensus_class = class_counts.index[0]
        consensus_votes = int(class_counts.iloc[0])
        total_votes = len(results)
        consensus_rate = (consensus_votes / total_votes) * 100

        prob_accum = {c: 0.0 for c in CLASS_NAMES}
        prob_count = 0
        for r in results:
            if r.get("probabilities"):
                prob_count += 1
                for c in CLASS_NAMES:
                    prob_accum[c] += r["probabilities"].get(c, 0.0)

        mean_probabilities = (
            {c: prob_accum[c] / prob_count for c in CLASS_NAMES}
            if prob_count > 0 else {}
        )

        consensus_summary = {
            "consensus_class": consensus_class,
            "consensus_votes": consensus_votes,
            "total_votes": total_votes,
            "consensus_rate": consensus_rate,
            "mean_probabilities": mean_probabilities,
            "agreement_status": (
                "Unanimous Consensus" if consensus_votes == total_votes
                else "Majority Agreement" if consensus_rate >= 60
                else "Split / Borderline Diagnosis"
            ),
        }
    else:
        consensus_summary = None

    return results, consensus_summary


@st.cache_data
def get_sample_library():
    """Discover sample test images organized by category."""
    samples = {}
    if TEST_DATASET_DIR.is_dir():
        for c in CLASS_NAMES:
            c_dir = TEST_DATASET_DIR / c
            if c_dir.is_dir():
                file_list = sorted(list(c_dir.glob("*.jpg")) + list(c_dir.glob("*.png")))
                if file_list:
                    samples[c] = file_list[:8]
    return samples


# --------------------------------------------------
# 6. MAIN APPLICATION & NAVIGATION
# --------------------------------------------------

def main():
    # Sidebar Navigation Menu
    st.sidebar.markdown("## 🧠 Navigation")
    nav_selection = st.sidebar.radio(
        "Select Application Module:",
        [
            "🏠 Home / Dashboard",
            "🔬 MRI Image Prediction",
            "⚖️ Model Comparison (CNN vs. SVM)",
            "📊 Model Performance Dashboard",
            "📁 Dataset Information",
            "ℹ️ About & Technical Details",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ System Status")
    models_dict, model_status = load_all_models()
    active_count = sum(1 for v in model_status.values() if v is True)
    total_count = len(model_status)
    st.sidebar.caption(f"**Active Engines**: {active_count} of {total_count} Models Loaded")

    # ==================================================
    # MODULE 1: HOME / DASHBOARD
    # ==================================================
    if nav_selection == "🏠 Home / Dashboard":
        st.markdown("<h1 class='main-title'>MRI Brain Tumor Detection</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p class='sub-title'>Medical AI Research Platform for Automated Brain Tumor Categorization via Deep Learning & Hybrid Transfer Models</p>",
            unsafe_allow_html=True,
        )

        # Overview Banner
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%); color: white; padding: 24px; border-radius: 14px; margin-bottom: 22px;">
                <h3 style="margin-top: 0; color: white;">Deep Learning & Classical ML Hybrid Architecture</h3>
                <p style="font-size: 1.05rem; opacity: 0.95; line-height: 1.6; margin-bottom: 12px;">
                    This system investigates automated detection and subtyping of brain neoplasms from axial, sagittal, and coronal MRI scans. 
                    By pairing ImageNet-pretrained <strong>MobileNetV2 feature representations (1280-d)</strong> with cross-validated 
                    <strong>Support Vector Classifiers (SVM)</strong> and <strong>Custom CNNs</strong>, the platform achieves robust, clinically actionable classification.
                </p>
                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                    <span style="background: rgba(255,255,255,0.2); padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">4 Pathologies</span>
                    <span style="background: rgba(255,255,255,0.2); padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">12,064 Curated Images</span>
                    <span style="background: rgba(255,255,255,0.2); padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">2,414 Independent Test Images</span>
                    <span style="background: rgba(255,255,255,0.2); padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">Top Test Accuracy: 92.50%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4 Supported Classes
        st.markdown("### 🔬 Supported Diagnostic Categories")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                """
                <div class="metric-card" style="border-top: 4px solid #D97706;">
                    <h4>⚠️ Glioma</h4>
                    <p style="font-size: 0.88rem; color: #64748B;">Malignant infiltrative tumor arising from glial cells (astrocytomas, glioblastomas). Often presents with irregular borders and central necrosis.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
                <div class="metric-card" style="border-top: 4px solid #7C3AED;">
                    <h4>🟣 Meningioma</h4>
                    <p style="font-size: 0.88rem; color: #64748B;">Typically benign extra-axial neoplasm arising from the arachnoid layer of meninges, frequently showing the characteristic dural tail sign.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                """
                <div class="metric-card" style="border-top: 4px solid #059669;">
                    <h4>✅ No Tumor</h4>
                    <p style="font-size: 0.88rem; color: #64748B;">Healthy brain parenchyma showing anatomical bilateral symmetry, normal ventricular contours, and preserved sulcal margins.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                """
                <div class="metric-card" style="border-top: 4px solid #0284C7;">
                    <h4>🔵 Pituitary</h4>
                    <p style="font-size: 0.88rem; color: #64748B;">Sellar and suprasellar neoplasms originating in pituitary gland tissue, often associated with optic chiasm compression and endocrine changes.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Model Performance Highlights
        st.markdown("### 🏆 Research Performance Highlights")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("MobileNetV2 + SVM", "92.50%", "Best Test Accuracy")
        with m2:
            st.metric("CNN + MobileNetV2 + SVM", "92.42%", "Feature Fusion (1408-d)")
        with m3:
            st.metric("MobileNetV2 + Logistic Reg.", "89.06%", "Fast Convex Baseline")
        with m4:
            st.metric("Custom CNN", "64.79%", "Standalone ConvNet")

        st.markdown("### 📊 Dataset Overview")
        st.write(
            "The model pipeline was trained and evaluated on **12,064 MRI scans** stratified into strict "
            "Training (7,720), Validation (1,930), and Held-Out Test (2,414) partitions."
        )

        stats_summary = pd.DataFrame([
            {"Partition": "Training Set", "Images": 7720, "Proportion": "64.0%"},
            {"Partition": "Validation Set", "Images": 1930, "Proportion": "16.0%"},
            {"Partition": "Held-Out Test Set", "Images": 2414, "Proportion": "20.0%"},
            {"Partition": "Total Corpus", "Images": 12064, "Proportion": "100.0%"},
        ])
        st.dataframe(stats_summary, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Proceed to MRI Image Prediction", type="primary"):
            st.session_state["nav_override"] = "🔬 MRI Image Prediction"
            st.rerun()

    # ==================================================
    # MODULE 2: MRI IMAGE PREDICTION
    # ==================================================
    elif nav_selection == "🔬 MRI Image Prediction":
        st.markdown("<h2 class='main-title'>🔬 MRI Image Prediction</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p class='sub-title'>Upload an MRI scan or select a verified clinical test scan to obtain multi-model categorization and probability scores.</p>",
            unsafe_allow_html=True,
        )

        sample_library = get_sample_library()
        has_samples = bool(sample_library)

        source_tab1, source_tab2 = st.tabs(["📁 Upload Your Own MRI Scan", "🧪 Select from Verified Clinical Test Library"])
        selected_image = None
        source_label = ""

        with source_tab1:
            uploaded_file = st.file_uploader(
                "Upload Brain MRI Scan (Supported formats: PNG, JPG, JPEG, BMP, WEBP)",
                type=["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"],
                key="uploader_input",
            )
            if uploaded_file is not None:
                # File validation
                if uploaded_file.size > 15 * 1024 * 1024:
                    st.error("Uploaded file exceeds maximum permissible size (15 MB).")
                else:
                    try:
                        selected_image = Image.open(uploaded_file)
                        source_label = f"Uploaded Scan: {uploaded_file.name}"
                    except Exception as e:
                        st.error(f"Invalid image format: {e}")

        with source_tab2:
            if has_samples:
                c_sel1, c_sel2 = st.columns(2)
                with c_sel1:
                    class_choice = st.selectbox(
                        "Select Pathology Category:",
                        CLASS_NAMES,
                        format_func=lambda c: f"{CLASS_ICONS[c]} {CLASS_DISPLAY_NAMES[c]}",
                    )
                with c_sel2:
                    sample_options = sample_library.get(class_choice, [])
                    sample_file = st.selectbox(
                        "Select Clinical Test Scan:",
                        sample_options,
                        format_func=lambda p: f"Held-out Scan #{p.stem} ({p.name})",
                    )
                if st.button("Load Selected Test Sample", type="secondary"):
                    if sample_file and sample_file.is_file():
                        selected_image = Image.open(sample_file)
                        source_label = f"Verified Test Set: {CLASS_DISPLAY_NAMES[class_choice]} ({sample_file.name})"
            else:
                st.info("Clinical test sample directory not detected.")

        if selected_image is not None:
            st.divider()
            col_view, col_diag = st.columns([1, 1.25], gap="large")

            with col_view:
                st.markdown("#### 🖼️ Radiological Image Preview")
                filter_choice = st.radio(
                    "Image Inspection Mode:",
                    ["Standard Original", "CLAHE Contrast Enhancement", "Edge Contours", "Inverted Lightbox"],
                    horizontal=True,
                )

                image_clean, cnn_input, mobilenet_input = prepare_image(selected_image)

                if filter_choice == "CLAHE Contrast Enhancement":
                    preview_img = apply_clahe_filter(image_clean)
                    cap_text = "CLAHE Adaptive Histogram Equalization"
                elif filter_choice == "Edge Contours":
                    preview_img = apply_edge_filter(image_clean)
                    cap_text = "Canny edge detection of lesion boundaries"
                elif filter_choice == "Inverted Lightbox":
                    preview_img = apply_inverted_filter(image_clean)
                    cap_text = "Inverted radiological lightbox view"
                else:
                    preview_img = image_clean
                    cap_text = "Standard preprocessed 224×224 RGB image"

                st.image(preview_img, caption=cap_text, use_container_width=True)
                st.caption(f"**Origin**: {source_label}")

                with st.expander("Detailed Image Specifications"):
                    st.write(f"- **Original Dimensions**: {selected_image.size[0]} × {selected_image.size[1]} pixels")
                    st.write(f"- **Model Input Dimensions**: 224 × 224 × 3")
                    st.write(f"- **Color Channel Mode**: {selected_image.mode}")
                    px_arr = np.asarray(image_clean)
                    st.write(f"- **Pixel Min / Max**: {px_arr.min()} / {px_arr.max()}")

            with col_diag:
                st.markdown("#### ⚡ AI Diagnostic Results")
                predict_trigger = st.button("Execute Diagnostic Inference", type="primary", use_container_width=True)

                results, consensus = run_all_predictions(models_dict, cnn_input, mobilenet_input)

                if results and consensus:
                    c_class = consensus["consensus_class"]
                    c_name = CLASS_DISPLAY_NAMES[c_class]
                    c_color = CLASS_COLORS[c_class]
                    c_bg = CLASS_BG_COLORS[c_class]
                    c_icon = CLASS_ICONS[c_class]

                    st.markdown(
                        f"""
                        <div class="consensus-banner" style="background: {c_bg}; border-left: 6px solid {c_color};">
                            <span style="color: {c_color}; font-weight: 700; text-transform: uppercase; font-size: 0.85rem;">Ensemble Consensus Verdict</span>
                            <h2 style="margin: 4px 0; color: #0F172A;">{c_icon} {c_name}</h2>
                            <p style="margin: 0; color: #475569; font-size: 0.95rem;">
                                <strong>{consensus['consensus_votes']} of {consensus['total_votes']} models agree</strong> ({consensus['consensus_rate']:.1f}% Agreement) — <em>{consensus['agreement_status']}</em>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Top model highlight
                    top_model = next((r for r in results if r["model"] == "MobileNetV2 + SVM"), results[0])
                    st.markdown(
                        f"**Top Ranked Model (MobileNetV2 + SVM)**: "
                        f"`{CLASS_DISPLAY_NAMES[top_model['predicted_class']]}` with "
                        f"**{top_model['confidence']:.2%}** confidence score."
                    )

                    # Class Probability Breakdown
                    if consensus["mean_probabilities"]:
                        st.markdown("**Ensemble Class Probability Distribution:**")
                        for c in CLASS_NAMES:
                            val = consensus["mean_probabilities"].get(c, 0.0)
                            col_txt, col_bar = st.columns([1.5, 3])
                            with col_txt:
                                st.caption(f"{CLASS_ICONS[c]} {CLASS_DISPLAY_NAMES[c]}: **{val:.1%}**")
                            with col_bar:
                                st.progress(min(max(float(val), 0.0), 1.0))

                    st.markdown("---")
                    st.markdown("##### Individual Classifier Breakdown:")
                    for r in results:
                        with st.container():
                            rc1, rc2 = st.columns([2, 1])
                            with rc1:
                                st.markdown(f"**{r['model']}** (`{r['benchmark_accuracy']}` test acc)")
                            with rc2:
                                p_c = r["predicted_class"]
                                p_conf = r["confidence"]
                                conf_str = f" ({p_conf:.1%})" if p_conf is not None else ""
                                st.markdown(
                                    f"<span class='badge-pill' style='background:{CLASS_BG_COLORS[p_c]}; color:{CLASS_COLORS[p_c]}; border: 1px solid {CLASS_COLORS[p_c]};'>"
                                    f"{CLASS_ICONS[p_c]} {CLASS_DISPLAY_NAMES[p_c]}{conf_str}</span>",
                                    unsafe_allow_html=True,
                                )

        st.markdown(
            """
            <div class="disclaimer-box">
                <strong>Medical Research Disclaimer:</strong> This software is an academic artificial intelligence demonstrator. 
                It is not an FDA-cleared diagnostic medical device and must never be utilized for primary patient diagnosis or treatment decisions without certified radiologist review.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ==================================================
    # MODULE 3: MODEL COMPARISON (CNN VS. MOBILENETV2 + SVM)
    # ==================================================
    elif nav_selection == "⚖️ Model Comparison (CNN vs. SVM)":
        st.markdown("<h2 class='main-title'>⚖️ Model Comparison: Custom CNN vs. MobileNetV2 + SVM</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p class='sub-title'>Direct side-by-side comparative analysis of standalone Deep Learning versus Hybrid Transfer Learning on identical MRI scans.</p>",
            unsafe_allow_html=True,
        )

        sample_library = get_sample_library()
        test_img_path = None

        comp_source = st.radio(
            "Select Scan for Comparison:",
            ["Choose from Verified Test Library", "Upload Custom Scan"],
            horizontal=True,
        )

        if comp_source == "Choose from Verified Test Library" and sample_library:
            c1, c2 = st.columns(2)
            with c1:
                cmp_cls = st.selectbox("Pathology Class:", CLASS_NAMES, format_func=lambda c: CLASS_DISPLAY_NAMES[c])
            with c2:
                cmp_sample = st.selectbox("Select Test Image:", sample_library.get(cmp_cls, []), format_func=lambda p: p.name)
            if cmp_sample and cmp_sample.is_file():
                comp_image = Image.open(cmp_sample)
        else:
            up_comp = st.file_uploader("Upload MRI Scan for Comparison:", type=["png", "jpg", "jpeg"])
            comp_image = Image.open(up_comp) if up_comp else None

        if comp_image is not None:
            _, cnn_in, mob_in = prepare_image(comp_image)
            st.divider()

            col_disp, col_res = st.columns([1, 2], gap="large")
            with col_disp:
                st.image(comp_image, caption="Target MRI Scan", use_container_width=True)

            with col_res:
                st.markdown("#### ⏱️ Real-Time Inference Benchmark")

                # 1. Custom CNN Execution
                cnn_model = models_dict["cnn"]
                if cnn_model is not None:
                    t0 = time.perf_counter()
                    cnn_probs = cnn_model.predict(cnn_in, verbose=0)[0]
                    cnn_time = (time.perf_counter() - t0) * 1000
                    cnn_idx = int(np.argmax(cnn_probs))
                    cnn_pred = class_name(cnn_idx)
                    cnn_score = float(cnn_probs[cnn_idx])
                else:
                    cnn_pred, cnn_score, cnn_time = "Unavailable", None, 0.0

                # 2. MobileNetV2 + SVM Execution
                mob_model = models_dict["mobilenet"]
                svm_model = models_dict["hybrid_models"].get("MobileNetV2 + SVM")

                if mob_model is not None and svm_model is not None:
                    t0 = time.perf_counter()
                    mob_feat = mob_model.predict(mob_in, verbose=0)
                    svm_label = int(svm_model.predict(mob_feat)[0])
                    svm_probs = svm_model.predict_proba(mob_feat)[0]
                    svm_time = (time.perf_counter() - t0) * 1000

                    svm_pred = class_name(svm_label)
                    score_map = {class_name(lbl): prob for lbl, prob in zip(svm_model.classes_, svm_probs)}
                    svm_score = float(score_map.get(svm_pred, 0.0))
                else:
                    svm_pred, svm_score, svm_time = "Unavailable", None, 0.0

                # Agreement Banner
                models_agree = (cnn_pred == svm_pred)
                if models_agree:
                    st.success(f"✅ **Models Agree**: Both classifiers identified **{CLASS_DISPLAY_NAMES[cnn_pred]}**")
                else:
                    st.warning(
                        f"⚠️ **Model Disagreement**: Custom CNN predicted **{CLASS_DISPLAY_NAMES[cnn_pred]}**, "
                        f"while MobileNetV2 + SVM predicted **{CLASS_DISPLAY_NAMES[svm_pred]}**"
                    )

                # Side by Side Cards
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.markdown(
                        f"""
                        <div class="metric-card" style="border-top: 4px solid #64748B;">
                            <h4>Custom CNN</h4>
                            <p style="color: #64748B; font-size: 0.85rem; margin-bottom: 8px;">3-Block ConvNet (Test Acc: 64.79%)</p>
                            <span class="badge-pill" style="background:{CLASS_BG_COLORS[cnn_pred]}; color:{CLASS_COLORS[cnn_pred]}; font-size: 1rem;">
                                {CLASS_ICONS[cnn_pred]} {CLASS_DISPLAY_NAMES[cnn_pred]}
                            </span>
                            <h5 style="margin-top: 12px; margin-bottom: 2px;">Confidence: {cnn_score:.2%}</h5>
                            <p style="font-size: 0.85rem; color: #475569;">Inference Latency: <strong>{cnn_time:.1f} ms</strong></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with sc2:
                    st.markdown(
                        f"""
                        <div class="metric-card" style="border-top: 4px solid #0284C7;">
                            <h4>MobileNetV2 + SVM (Best)</h4>
                            <p style="color: #64748B; font-size: 0.85rem; margin-bottom: 8px;">Pretrained Features + Tuned RBF (Test Acc: 92.50%)</p>
                            <span class="badge-pill" style="background:{CLASS_BG_COLORS[svm_pred]}; color:{CLASS_COLORS[svm_pred]}; font-size: 1rem;">
                                {CLASS_ICONS[svm_pred]} {CLASS_DISPLAY_NAMES[svm_pred]}
                            </span>
                            <h5 style="margin-top: 12px; margin-bottom: 2px;">Confidence: {svm_score:.2%}</h5>
                            <p style="font-size: 0.85rem; color: #475569;">Inference Latency: <strong>{svm_time:.1f} ms</strong></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # ==================================================
    # MODULE 4: MODEL PERFORMANCE DASHBOARD
    # ==================================================
    elif nav_selection == "📊 Model Performance Dashboard":
        st.markdown("<h2 class='main-title'>📊 Model Performance Dashboard</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p class='sub-title'>Empirical evaluation results measured on the independent held-out test split (2,414 images).</p>",
            unsafe_allow_html=True,
        )

        # Overview Table
        comp_csv = OUTPUT_DIR / "all_model_comparison.csv"
        hybrid_csv = OUTPUT_DIR / "hybrid_model_comparison.csv"

        if hybrid_csv.is_file():
            df_hybrid = pd.read_csv(hybrid_csv)
            st.markdown("#### 📋 Detailed Hybrid Models Performance Table")

            formatted_df = df_hybrid.copy()
            for col in ["validation_accuracy", "test_accuracy", "weighted_precision", "weighted_recall", "weighted_f1"]:
                if col in formatted_df.columns:
                    formatted_df[col] = formatted_df[col].map(lambda x: f"{x:.2%}")

            formatted_df.columns = [
                c.replace("_", " ").title() for c in formatted_df.columns
            ]
            st.dataframe(formatted_df, use_container_width=True, hide_index=True)

        if comp_csv.is_file():
            df_comp = pd.read_csv(comp_csv)
            st.markdown("#### 📊 Comparative Model Accuracy Benchmark")
            chart_df = df_comp.copy()
            chart_df["Accuracy (%)"] = chart_df["Test accuracy"] * 100
            st.bar_chart(chart_df.set_index("Model")["Accuracy (%)"])

        st.divider()
        st.markdown("#### 🎯 Confusion Matrix & Classification Report Viewer")

        cm_dict = {
            "MobileNetV2 + SVM (Best Model)": (
                OUTPUT_DIR / "mobilenetv2_plus_svm_confusion_matrix.png",
                OUTPUT_DIR / "mobilenetv2_plus_svm_report.txt",
            ),
            "Combined CNN + MobileNetV2 + SVM": (
                OUTPUT_DIR / "combined_hybrid_confusion_matrix.png",
                OUTPUT_DIR / "combined_hybrid_report.txt",
            ),
            "MobileNetV2 + Logistic Regression": (
                OUTPUT_DIR / "mobilenetv2_plus_logistic_regression_confusion_matrix.png",
                OUTPUT_DIR / "mobilenetv2_plus_logistic_regression_report.txt",
            ),
            "MobileNetV2 + Random Forest": (
                OUTPUT_DIR / "mobilenetv2_plus_random_forest_confusion_matrix.png",
                OUTPUT_DIR / "mobilenetv2_plus_random_forest_report.txt",
            ),
            "Custom CNN": (
                OUTPUT_DIR / "cnn_confusion_matrix.png",
                OUTPUT_DIR / "cnn_classification_report.txt",
            ),
        }

        col_cm_sel, col_cm_view = st.columns([1, 1.4], gap="large")
        with col_cm_sel:
            chosen_model = st.selectbox("Select Model Architecture:", list(cm_dict.keys()))
            _, rep_path = cm_dict[chosen_model]
            if rep_path.is_file():
                st.markdown("**Classification Metrics Report:**")
                st.code(rep_path.read_text(encoding="utf-8"))

        with col_cm_view:
            cm_img_path, _ = cm_dict[chosen_model]
            if cm_img_path.is_file():
                st.image(str(cm_img_path), caption=f"{chosen_model} — Test Confusion Matrix", use_container_width=True)

    # ==================================================
    # MODULE 5: DATASET INFORMATION
    # ==================================================
    elif nav_selection == "📁 Dataset Information":
        st.markdown("<h2 class='main-title'>📁 Dataset Information & Statistics</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p class='sub-title'>Detailed breakdown of the 12,064 MRI image dataset across classes and data partitions.</p>",
            unsafe_allow_html=True,
        )

        dataset_stats = pd.DataFrame([
            {"Class": "Glioma", "Training": 2414, "Validation": 604, "Test": 755, "Total": 3773},
            {"Class": "Meningioma", "Training": 1747, "Validation": 436, "Test": 546, "Total": 2729},
            {"Class": "No Tumor", "Training": 1556, "Validation": 389, "Test": 487, "Total": 2432},
            {"Class": "Pituitary", "Training": 2003, "Validation": 501, "Test": 626, "Total": 3130},
            {"Class": "Total", "Training": 7720, "Validation": 1930, "Test": 2414, "Total": 12064},
        ])

        st.markdown("#### 📊 Stratified Distribution Table (Verified Against Manifests)")
        st.dataframe(dataset_stats, use_container_width=True, hide_index=True)

        # Visual Class Chart
        st.markdown("#### 📈 Class Distribution Across Dataset")
        plot_df = dataset_stats[dataset_stats["Class"] != "Total"].set_index("Class")[["Training", "Validation", "Test"]]
        st.bar_chart(plot_df)

        st.markdown("#### 🔍 Dataset Preprocessing Rules")
        st.markdown(
            """
            - **Input Resolution**: Resized with Lanczos interpolation to `224 × 224` pixels.
            - **Color Channels**: Transposed from EXIF and validated as 3-channel RGB.
            - **Normalizations**:
              - Standalone CNN: $$I_{\\text{CNN}} = \\frac{I}{255.0} \\in [0, 1]$$
              - MobileNetV2: $$I_{\\text{MobileNet}} = \\frac{I}{127.5} - 1.0 \\in [-1, 1]$$
            - **Partitioning Protocol**: 80% Train-Val split (stratified into 80% train, 20% validation) with a completely isolated 20% held-out test split.
            """
        )

    # ==================================================
    # MODULE 6: ABOUT / TECHNICAL INFORMATION
    # ==================================================
    elif nav_selection == "ℹ️ About & Technical Details":
        st.markdown("<h2 class='main-title'>ℹ️ Technical Architecture & Stack</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p class='sub-title'>Comprehensive technical reference of frameworks, deep learning models, and algorithms powering the suite.</p>",
            unsafe_allow_html=True,
        )

        tech_specs = [
            ("Python (3.11)", "Core programming language providing modern typing, performance optimizations, and native compatibility with scientific ML libraries."),
            ("TensorFlow & Keras", "Industrial deep learning framework utilized for model definition, tensor pipelines, pretrained MobileNetV2 integration, and GPU/CPU acceleration."),
            ("MobileNetV2", "Lightweight, highly efficient deep CNN utilizing inverted residual blocks and depthwise separable convolutions. Pretrained on ImageNet and employed with Global Average Pooling (1280-d feature vector)."),
            ("Custom CNN", "3-block standalone Convolutional Neural Network (32, 64, 128 filters) with MaxPooling, Dropout regularization, and a 128-unit dense embedding layer."),
            ("Support Vector Classifier (SVM)", "Max-margin classifier equipped with a Radial Basis Function (RBF) kernel. Optimized via 3-fold Stratified GridSearchCV with fold-isolated StandardScaler transformation."),
            ("Scikit-learn", "Machine learning library powering the Pipeline abstractions, GridSearchCV hyperparameter search, Random Forest, Logistic Regression, and evaluation metrics."),
            ("NumPy & Pandas", "Foundation for high-dimensional feature caching (.npz format), tensor manipulation, dataset manifests, and tabular performance logging."),
            ("Matplotlib & Seaborn", "Statistical plotting libraries used for rendering publication-grade confusion matrices and training history curves."),
            ("Streamlit", "Modern reactive web application framework powering this clinical AI research dashboard."),
        ]

        for title, desc in tech_specs:
            with st.container(border=True):
                st.markdown(f"#### ⚙️ {title}")
                st.write(desc)


if __name__ == "__main__":
    main()
