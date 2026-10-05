"""
CropGuard AI - Disease Database Manager
Provides structured access to symptoms, treatments, prevention, and weather triggers
for all 38 supported crop disease classes.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

# Path to persistent database JSON
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "disease_database.json"

# All 38 model class names in exact index order matching trained model
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

_DB_CACHE: Optional[Dict[str, Any]] = None


def load_database() -> Dict[str, Any]:
    """Load the JSON database into memory with caching."""
    global _DB_CACHE
    if _DB_CACHE is not None:
        return _DB_CACHE

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Disease database not found at: {DB_PATH}")

    with open(DB_PATH, "r", encoding="utf-8") as f:
        _DB_CACHE = json.load(f)

    return _DB_CACHE


def get_disease_info(class_name: str) -> Dict[str, Any]:
    """Retrieve full advisory record for a given class name.
    Falls back to a well-structured generic response if not found.
    """
    db = load_database()
    if class_name in db:
        info = db[class_name].copy()
        info["raw_class"] = class_name
        return info

    # Fallback parsing
    parts = class_name.split("___")
    crop = parts[0].replace("_", " ")
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    is_healthy = "healthy" in disease.lower()

    return {
        "crop": crop,
        "disease": disease,
        "status": "Healthy" if is_healthy else "Diseased",
        "severity": "None" if is_healthy else "Moderate",
        "pathogen": "N/A" if is_healthy else "Under Investigation",
        "description": f"Diagnosis for {crop} condition: {disease}.",
        "symptoms": ["Normal appearance" if is_healthy else "Abnormal discoloration or lesions noticed on leaf tissue."],
        "organic_treatment": ["Maintain balanced hydration and nutrition."],
        "chemical_treatment": ["Consult local agricultural extension service for registered products."],
        "prevention": ["Practice regular monitoring, proper irrigation, and canopy management."],
        "favorable_weather": "Varies by local microclimate conditions.",
        "raw_class": class_name
    }


def get_all_crops() -> List[str]:
    """Return sorted unique list of all crops represented in database."""
    db = load_database()
    crops = sorted(list({v.get("crop", "Unknown") for v in db.values()}))
    return crops


def get_records_by_crop(crop_name: str) -> List[Dict[str, Any]]:
    """Return all disease records for a specific crop."""
    db = load_database()
    return [
        {**record, "class_key": key}
        for key, record in db.items()
        if record.get("crop", "").lower() == crop_name.lower()
    ]


def search_database(query: str) -> List[Dict[str, Any]]:
    """Search diseases across crop names, disease names, symptoms, and pathogens."""
    db = load_database()
    q = query.strip().lower()
    if not q:
        return [{**v, "class_key": k} for k, v in db.items()]

    results = []
    for k, v in db.items():
        text_corpus = (
            f"{v.get('crop', '')} {v.get('disease', '')} {v.get('pathogen', '')} "
            f"{v.get('description', '')} {' '.join(v.get('symptoms', []))}"
        ).lower()
        if q in text_corpus:
            results.append({**v, "class_key": k})
    return results


def format_markdown_report(disease_info: Dict[str, Any], confidence: float) -> str:
    """Generate a clean printable/downloadable Markdown report of diagnosis and advisory."""
    crop = disease_info.get("crop", "Crop")
    disease = disease_info.get("disease", "Condition")
    status = disease_info.get("status", "Unknown")
    pathogen = disease_info.get("pathogen", "N/A")
    severity = disease_info.get("severity", "N/A")
    weather = disease_info.get("favorable_weather", "N/A")

    symptoms_md = "\n".join([f"- {s}" for s in disease_info.get("symptoms", [])])
    organic_md = "\n".join([f"- {t}" for t in disease_info.get("organic_treatment", [])])
    chemical_md = "\n".join([f"- {t}" for t in disease_info.get("chemical_treatment", [])])
    prevention_md = "\n".join([f"- {p}" for p in disease_info.get("prevention", [])])

    return f"""# 🌿 CropGuard AI — Agronomic Diagnostic Report

**Diagnosis:** {crop} — {disease}  
**Status:** {status}  
**Confidence Score:** {confidence * 100:.2f}%  
**Pathogen / Agent:** {pathogen}  
**Severity Index:** {severity}  
**Conducive Weather:** {weather}  

---

### 🔍 Observed Symptoms & Identification
{symptoms_md}

---

### 🍃 Organic & Cultural Controls
{organic_md}

---

### 🧪 Chemical & Therapeutic Treatments
{chemical_md}

---

### 🛡️ Long-term Preventive Management
{prevention_md}

---
*Report generated by CropGuard AI Diagnostic & Advisory System.*
"""
