"""
CropGuard AI - Deep Learning Model Inference Engine
Handles loading, preprocessing, caching, and classification for plant disease diagnosis.
"""

from pathlib import Path
from typing import Dict, Any, List, Union
import numpy as np
from PIL import Image

# Path to trained model
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "trained_plant_disease_model.h5"

# The 38 classes matching trained weights index order
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

_CACHED_MODEL = None


def load_classification_model():
    """Load and cache the pre-trained Keras model."""
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    import tensorflow as tf
    # Load model
    _CACHED_MODEL = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    return _CACHED_MODEL


def preprocess_image(image_input: Union[str, Path, Image.Image, bytes]) -> np.ndarray:
    """Preprocess image input to target tensor shape (1, 128, 128, 3) in RGB format."""
    if isinstance(image_input, (str, Path)):
        img = Image.open(str(image_input))
    elif isinstance(image_input, bytes):
        import io
        img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, Image.Image):
        img = image_input
    elif hasattr(image_input, "read"):
        import io
        img = Image.open(image_input)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Ensure RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize to exact dimensions expected by trained model (128x128)
    img = img.resize((128, 128), Image.Resampling.BILINEAR)

    # Convert to array (float32, 0-255 scale as trained with tf.keras.preprocessing)
    img_array = np.array(img, dtype=np.float32)

    # Add batch dimension: (1, 128, 128, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch


def predict_disease(image_input, top_k: int = 4) -> Dict[str, Any]:
    """Run model inference and return top predictions with confidence scores."""
    model = load_classification_model()
    processed_tensor = preprocess_image(image_input)

    raw_preds = model.predict(processed_tensor, verbose=0)[0]

    # Convert to probabilities if necessary
    if raw_preds.min() < 0 or raw_preds.max() > 1.0 or not np.isclose(np.sum(raw_preds), 1.0, atol=1e-2):
        exp_preds = np.exp(raw_preds - np.max(raw_preds))
        probabilities = exp_preds / np.sum(exp_preds)
    else:
        probabilities = raw_preds

    top_indices = np.argsort(probabilities)[::-1][:top_k]

    primary_idx = int(top_indices[0])
    primary_class = CLASS_NAMES[primary_idx]
    primary_confidence = float(probabilities[primary_idx])

    top_predictions: List[Dict[str, Any]] = []
    for idx in top_indices:
        cls_name = CLASS_NAMES[int(idx)]
        score = float(probabilities[int(idx)])
        # Parse clean names
        parts = cls_name.split("___")
        crop = parts[0].replace("_", " ")
        cond = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
        top_predictions.append({
            "class_name": cls_name,
            "crop": crop,
            "condition": cond,
            "confidence": score,
            "percentage": f"{score * 100:.1f}%"
        })

    return {
        "class_name": primary_class,
        "class_index": primary_idx,
        "confidence": primary_confidence,
        "confidence_percentage": f"{primary_confidence * 100:.2f}%",
        "top_predictions": top_predictions,
        "is_confident": primary_confidence >= 0.50
    }
