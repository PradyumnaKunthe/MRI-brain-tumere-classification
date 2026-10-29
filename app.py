
"""MRI Brain Tumor Detection - Model Comparison Dashboard."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image, ImageOps


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

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

st.set_page_config(
    page_title="MRI Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
)


# --------------------------------------------------
# 2. LOAD ALL SAVED MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():
    """Load the trained models without retraining them."""

    required_files = [
        CNN_PATH,
        MOBILENET_PATH,
        COMBINED_SVM_PATH,
        *HYBRID_PATHS.values(),
    ]

    missing = [str(p) for p in required_files if not p.is_file()]

    if missing:
        raise FileNotFoundError(
            "Missing required model files:\n" + "\n".join(missing)
        )

    cnn = tf.keras.models.load_model(CNN_PATH, compile=False)

    # The custom CNN's 128-unit dense layer is its learned embedding.
    cnn_feature_model = tf.keras.Model(
        inputs=cnn.input,
        outputs=cnn.get_layer("dense").output,
        name="custom_cnn_feature_extractor",
    )

    mobilenet = tf.keras.models.load_model(
        MOBILENET_PATH,
        compile=False,
    )

    combined_svm = joblib.load(COMBINED_SVM_PATH)

    hybrid_models = {
        name: joblib.load(path)
        for name, path in HYBRID_PATHS.items()
    }

    return cnn, cnn_feature_model, mobilenet, combined_svm, hybrid_models


# --------------------------------------------------
# 3. IMAGE PREPROCESSING
# --------------------------------------------------

def prepare_image(pil_image):
    """Create input arrays matching each model's training pipeline."""

    image = ImageOps.exif_transpose(pil_image).convert("RGB")
    image = image.resize(
        (224, 224),
        Image.Resampling.LANCZOS,
    )

    pixels = np.asarray(image, dtype=np.float32)

    # Custom CNN input: pixel range [0, 1].
    cnn_input = np.expand_dims(pixels / 255.0, axis=0)

    # MobileNetV2 input: pixel range [-1, 1].
    mobilenet_input = np.expand_dims(
        pixels / 127.5 - 1.0,
        axis=0,
    )

    return image, cnn_input, mobilenet_input


# --------------------------------------------------
# 4. PREDICTION HELPERS
# --------------------------------------------------

def class_name(label):
    index = int(label)

    if not 0 <= index < len(CLASS_NAMES):
        raise ValueError(f"Unexpected class index: {index}")

    return CLASS_NAMES[index]


def predict_with_score(model, features):
    """Predict one class and return a score where available."""

    predicted_label = int(model.predict(features)[0])
    predicted_class = class_name(predicted_label)
    score = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]

        # Map probability columns to their class labels.
        score_by_label = {
            int(label): float(probability)
            for label, probability in zip(
                model.classes_,
                probabilities,
            )
        }

        score = score_by_label.get(predicted_label)

    return predicted_class, score


def predict_all(
    cnn,
    cnn_feature_model,
    mobilenet,
    combined_svm,
    hybrid_models,
    cnn_input,
    mobilenet_input,
):
    results = []

    # 1. Standalone CNN.
    cnn_probabilities = cnn.predict(cnn_input, verbose=0)[0]
    cnn_index = int(np.argmax(cnn_probabilities))

    results.append({
        "Model": "Custom CNN",
        "Predicted class": class_name(cnn_index),
        "Model-reported score": float(cnn_probabilities[cnn_index]),
    })

    # Get both feature vectors for the same uploaded image.
    cnn_features = cnn_feature_model.predict(
        cnn_input,
        verbose=0,
    )

    mobilenet_features = mobilenet.predict(
        mobilenet_input,
        verbose=0,
    )

    # The combined SVM was trained on 128 + 1280 = 1408 features.
    combined_features = np.concatenate(
        [cnn_features, mobilenet_features],
        axis=1,
    ).astype(np.float32)

    if combined_features.shape[1] != 1408:
        raise ValueError(
            "Unexpected combined feature size: "
            f"{combined_features.shape[1]}, expected 1408."
        )

    # 2. Combined CNN + MobileNetV2 + SVM.
    # This SVM was trained without probability calibration.
    combined_class = int(combined_svm.predict(combined_features)[0])

    results.append({
        "Model": "Combined CNN + MobileNetV2 + SVM",
        "Predicted class": class_name(combined_class),
        "Model-reported score": None,
    })

    # 3–5. Individual MobileNetV2 hybrid classifiers.
    for name, model in hybrid_models.items():
        predicted_class, score = predict_with_score(
            model,
            mobilenet_features,
        )

        results.append({
            "Model": name,
            "Predicted class": predicted_class,
            "Model-reported score": score,
        })

    return results


# --------------------------------------------------
# 5. DISPLAY MEASURED TEST ACCURACIES
# --------------------------------------------------

def show_test_metrics():
    """Display previously measured held-out test results."""

    rows = []

    cnn_metrics = OUTPUT_DIR / "cnn_metrics.json"
    hybrid_metrics = OUTPUT_DIR / "hybrid_model_metrics.json"
    combined_metrics = OUTPUT_DIR / "combined_hybrid_metrics.json"

    if cnn_metrics.exists():
        with cnn_metrics.open(encoding="utf-8") as file:
            data = json.load(file)

        rows.append({
            "Model": data["model"],
            "Test accuracy": float(data["test_accuracy"]),
        })

    if hybrid_metrics.exists():
        with hybrid_metrics.open(encoding="utf-8") as file:
            for item in json.load(file):
                rows.append({
                    "Model": item["model"],
                    "Test accuracy": float(item["test_accuracy"]),
                })

    if combined_metrics.exists():
        with combined_metrics.open(encoding="utf-8") as file:
            data = json.load(file)

        rows.append({
            "Model": data["model"],
            "Test accuracy": float(data["test_accuracy"]),
        })

    if not rows:
        st.info("Saved model evaluation metrics are not available.")
        return

    df = pd.DataFrame(rows).drop_duplicates(
        subset=["Model"],
        keep="last",
    )

    # Highlight the combined system against MobileNetV2 + SVM.
    scores = dict(zip(df["Model"], df["Test accuracy"]))

    cnn_score = scores.get("Custom CNN")
    svm_score = scores.get("MobileNetV2 + SVM")
    combined_score = scores.get("Combined CNN + MobileNetV2 + SVM")

    if cnn_score is not None:
        st.metric("Custom CNN", f"{cnn_score:.2%}")

    col1, col2 = st.columns(2)

    if svm_score is not None:
        col1.metric(
            "MobileNetV2 + SVM",
            f"{svm_score:.2%}",
        )

    if combined_score is not None:
        col2.metric(
            "Combined CNN + MobileNetV2 + SVM",
            f"{combined_score:.2%}",
        )

    if svm_score is not None and combined_score is not None:
        difference = (combined_score - svm_score) * 100
        st.caption(
            f"Combined model minus MobileNetV2 + SVM: "
            f"{difference:+.2f} percentage points."
        )

    st.dataframe(
        df.assign(
            **{
                "Test accuracy": df["Test accuracy"].map(
                    lambda x: f"{x:.2%}"
                )
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "These are saved test-set results, not predictions for the "
        "currently uploaded image."
    )


# --------------------------------------------------
# 6. STREAMLIT INTERFACE
# --------------------------------------------------

st.title("🧠 MRI Brain Tumor Detection")
st.write(
    "Compare a standalone CNN, a fused CNN + MobileNetV2 + SVM, "
    "and three individual MobileNetV2 hybrid classifiers."
)

st.warning(
    "Educational research prototype. This application is not a "
    "clinically validated diagnostic tool and must not be used "
    "for medical decisions."
)

with st.expander("Model evaluation results", expanded=True):
    show_test_metrics()

uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"],
)

if uploaded_file is not None:
    try:
        original = Image.open(uploaded_file)
        image, cnn_input, mobilenet_input = prepare_image(original)

        st.subheader("Uploaded MRI")
        st.image(
            image,
            caption="Preprocessed image used for inference",
            width=300,
        )

        if st.button(
            "Compare all models",
            type="primary",
            width="stretch",
        ):
            try:
                with st.spinner("Loading saved models and predicting..."):
                    (
                        cnn,
                        cnn_feature_model,
                        mobilenet,
                        combined_svm,
                        hybrid_models,
                    ) = load_models()

                    results = predict_all(
                        cnn,
                        cnn_feature_model,
                        mobilenet,
                        combined_svm,
                        hybrid_models,
                        cnn_input,
                        mobilenet_input,
                    )

                st.subheader("Predictions for this MRI")

                for start in range(0, len(results), 2):
                    columns = st.columns(2)

                    for offset, column in enumerate(columns):
                        index = start + offset

                        if index >= len(results):
                            break

                        result = results[index]

                        with column:
                            with st.container(border=True):
                                st.markdown(f"### {result['Model']}")
                                readable_class = result[
                                    "Predicted class"
                                ].replace("notumor", "No tumor").title()

                                st.metric(
                                    "Predicted class",
                                    readable_class,
                                )

                                score = result["Model-reported score"]

                                if score is not None:
                                    st.write(
                                        f"Model-reported score: {score:.2%}"
                                    )
                                    st.progress(
                                        float(np.clip(score, 0.0, 1.0))
                                    )
                                else:
                                    st.caption(
                                        "No probability score is configured "
                                        "for this SVM."
                                    )

                display_df = pd.DataFrame(results)
                display_df["Predicted class"] = (
                    display_df["Predicted class"].replace({
                        "notumor": "No tumor"
                    })
                )
                display_df["Model-reported score"] = (
                    display_df["Model-reported score"].map(
                        lambda value: (
                            f"{value:.2%}"
                            if value is not None
                            else "Not available"
                        )
                    )
                )

                st.subheader("Prediction summary")
                st.dataframe(
                    display_df,
                    width="stretch",
                    hide_index=True,
                )

                st.caption(
                    "Model-reported scores are not necessarily calibrated "
                    "confidence estimates. The combined SVM reports only "
                    "the predicted class in this implementation."
                )

            except Exception as error:
                st.error(
                    "Prediction failed. Verify that all model files exist "
                    "and that the saved models are compatible."
                )
                st.exception(error)

    except Exception as error:
        st.error("Could not read the uploaded image.")
        st.exception(error)

st.divider()
st.caption(
    "Models: Custom CNN | Combined CNN + MobileNetV2 + SVM | "
    "MobileNetV2 + SVM | MobileNetV2 + Random Forest | "
    "MobileNetV2 + Logistic Regression"
)
