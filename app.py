"""
CropGuard AI - Streamlit Web Application
Intelligent Plant Disease Detection & Agricultural Advisory System
"""

import os
from pathlib import Path
from PIL import Image
import streamlit as st
import pandas as pd

from backend.model import predict_disease, load_classification_model, CLASS_NAMES
from backend.database import (
    load_database,
    get_disease_info,
    get_all_crops,
    get_records_by_crop,
    search_database,
    format_markdown_report
)
from backend.advisory import get_ai_agronomic_consultation

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="CropGuard AI — Crop Disease Diagnosis & Advisory",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #424242;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        margin-bottom: 12px;
    }
    .healthy-banner {
        background: #e8f5e9;
        border-left: 6px solid #2e7d32;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 18px;
    }
    .disease-banner {
        background: #ffebee;
        border-left: 6px solid #c62828;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 18px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-danger { background-color: #ffcdd2; color: #b71c1c; }
    .badge-warning { background-color: #fff9c4; color: #f57f17; }
    .badge-success { background-color: #c8e6c9; color: #1b5e20; }
    .badge-info { background-color: #bbdefb; color: #0d47a1; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre-wrap;
        background-color: #f1f8e9;
        border-radius: 6px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2e7d32 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Model Warmup (Cached for fast instant predictions)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Initializing CropGuard Deep Learning Model...")
def get_cached_model():
    return load_classification_model()


# Warm up model at startup
try:
    _ = get_cached_model()
    MODEL_READY = True
except Exception as e:
    MODEL_READY = False
    MODEL_ERR = str(e)


# ---------------------------------------------------------
# Sidebar Navigation & Settings
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1592417817098-8f3d6910985c?w=500&auto=format&fit=crop&q=80", use_container_width=True)
    st.title("🌿 CropGuard AI")
    st.markdown("**Intelligent Crop Health & Pathology Advisory**")
    st.divider()

    nav_selection = st.radio(
        "Navigation",
        [
            "🩺 Crop Disease Diagnosis",
            "📚 Disease Database (38 Classes)",
            "🧠 Model & Architecture",
            "👨‍🌾 Agronomy Best Practices"
        ],
        index=0
    )

    st.divider()

    with st.expander("⚙️ AI Advisory Settings", expanded=False):
        # Resolve API key: Streamlit Cloud secrets > env var > user input
        _default_key = ""
        try:
            _default_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
        except Exception:
            _default_key = os.getenv("GEMINI_API_KEY", "")

        api_key_input = st.text_input(
            "Google Gemini API Key (Optional)",
            type="password",
            value=_default_key,
            help="Enables interactive live agronomist chat. If omitted, built-in offline agronomist engine will be used automatically!"
        )
        conf_threshold = st.slider(
            "Confidence Alert Threshold (%)",
            min_value=20,
            max_value=90,
            value=40,
            help="Alerts if top prediction certainty is lower than this value"
        )

    st.markdown("---")
    st.markdown("""
    **System Status:**  
    ✅ **Model:** 10-Layer CNN (Trained)  
    ✅ **Database:** 38 Classes (Offline JSON)  
    ✅ **Engine:** Active  
    """)
    st.caption("CropGuard AI v2.0 • Precision Agriculture")


# =========================================================
# PAGE 1: DIAGNOSIS HUB
# =========================================================
if nav_selection == "🩺 Crop Disease Diagnosis":
    st.markdown("<div class='main-header'>🩺 Plant Disease Diagnosis & Treatment Advisory</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Upload an image of a plant leaf or pick a test sample to identify diseases and receive expert treatment plans.</div>", unsafe_allow_html=True)

    input_mode = st.radio(
        "Choose Image Input Method:",
        ["📁 Upload Image", "📸 Camera Capture", "🖼️ Test Sample Gallery"],
        horizontal=True
    )

    selected_image = None
    image_source_name = ""

    if input_mode == "📁 Upload Image":
        uploaded_file = st.file_uploader(
            "Upload leaf photograph (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="For best accuracy, upload a clear photo of an individual leaf under good lighting."
        )
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file)
            image_source_name = uploaded_file.name

    elif input_mode == "📸 Camera Capture":
        cam_file = st.camera_input("Take a photo of the plant leaf")
        if cam_file is not None:
            selected_image = Image.open(cam_file)
            image_source_name = "Camera Capture"

    elif input_mode == "🖼️ Test Sample Gallery":
        samples_dir = Path("samples")
        if samples_dir.exists():
            sample_files = sorted([f.name for f in samples_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
            
            # Prioritize user uploaded and common test samples
            sample_choice = st.selectbox(
                "Select a sample image from the test repository:",
                sample_files,
                index=0 if sample_files else None
            )
            if sample_choice:
                sample_path = samples_dir / sample_choice
                selected_image = Image.open(sample_path)
                image_source_name = sample_choice

    st.divider()

    # If an image has been selected
    if selected_image is not None:
        col_img, col_diag = st.columns([1, 1.4], gap="large")

        with col_img:
            st.markdown(f"#### 📷 Inspected Leaf: `{image_source_name}`")
            st.image(selected_image, use_container_width=True, caption=f"Resolution: {selected_image.size[0]}x{selected_image.size[1]} | Mode: {selected_image.mode}")
            
            diagnose_btn = st.button("🚀 Diagnose Crop Pathology", type="primary", use_container_width=True)

        with col_diag:
            if diagnose_btn or "last_diagnosis" in st.session_state and st.session_state.get("last_image_name") == image_source_name:
                with st.spinner("Analyzing foliar features using Deep CNN Model..."):
                    # Run Model Prediction
                    pred_res = predict_disease(selected_image)
                    class_name = pred_res["class_name"]
                    confidence = pred_res["confidence"]
                    disease_info = get_disease_info(class_name)

                    # Cache in session state
                    st.session_state["last_diagnosis"] = {
                        "pred": pred_res,
                        "info": disease_info
                    }
                    st.session_state["last_image_name"] = image_source_name

            if "last_diagnosis" in st.session_state and st.session_state.get("last_image_name") == image_source_name:
                diag = st.session_state["last_diagnosis"]
                pred_res = diag["pred"]
                disease_info = diag["info"]
                class_name = pred_res["class_name"]
                confidence = pred_res["confidence"]
                is_healthy = disease_info.get("status") == "Healthy"

                # Status Banner
                if is_healthy:
                    st.markdown(f"""
                    <div class='healthy-banner'>
                        <h2 style='color:#1b5e20; margin:0;'>✅ Healthy Plant Detected</h2>
                        <p style='margin:4px 0 0 0; color:#2e7d32; font-weight:600;'>
                            {disease_info.get('crop')} shows no signs of active disease pathogens.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    severity = disease_info.get("severity", "Moderate")
                    badge_class = "badge-danger" if "High" in severity or "Severe" in severity else "badge-warning"
                    st.markdown(f"""
                    <div class='disease-banner'>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <h2 style='color:#b71c1c; margin:0;'>⚠️ Pathology Detected: {disease_info.get('disease')}</h2>
                            <span class='badge {badge_class}'>Severity: {severity}</span>
                        </div>
                        <p style='margin:6px 0 0 0; color:#424242;'>
                            <strong>Crop:</strong> {disease_info.get('crop')} &nbsp;|&nbsp; 
                            <strong>Pathogen:</strong> {disease_info.get('pathogen')}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Confidence Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Model Confidence", f"{confidence * 100:.1f}%")
                m2.metric("Plant Species", disease_info.get("crop", "Unknown"))
                m3.metric("Pathology Status", disease_info.get("status", "Unknown"))

                if confidence * 100 < conf_threshold:
                    st.warning(f"⚠️ Confidence score ({confidence * 100:.1f}%) is below your alert threshold ({conf_threshold}%). Inspect leaf margins or retake under clearer lighting.")

                # Probability Breakdown
                with st.expander("📊 Differential Diagnosis (Top Model Predictions)", expanded=False):
                    prob_df = pd.DataFrame([
                        {"Condition": f"{p['crop']} - {p['condition']}", "Probability": p["confidence"]}
                        for p in pred_res["top_predictions"]
                    ])
                    st.bar_chart(prob_df.set_index("Condition"))

        # Detailed Advisory Tabs
        if "last_diagnosis" in st.session_state and st.session_state.get("last_image_name") == image_source_name:
            diag = st.session_state["last_diagnosis"]
            disease_info = diag["info"]
            confidence = diag["pred"]["confidence"]

            st.markdown("### 📋 Agronomic Treatment & Advisory Plan")

            tab_sym, tab_org, tab_chem, tab_prev, tab_chat = st.tabs([
                "🔍 Symptoms & Biology",
                "🍃 Organic Remedies",
                "🧪 Chemical Treatments",
                "🛡️ Field Prevention",
                "💬 AI Agronomist Consultation"
            ])

            with tab_sym:
                st.markdown(f"**Pathogen / Causal Agent:** `{disease_info.get('pathogen', 'N/A')}`")
                st.markdown(f"**Description:** {disease_info.get('description', '')}")
                st.markdown("**Key Symptoms to Verify in the Field:**")
                for s in disease_info.get("symptoms", []):
                    st.markdown(f"- 🔎 {s}")
                st.info(f"🌦️ **Favorable Microclimate Trigger:** {disease_info.get('favorable_weather', 'Not specified')}")

            with tab_org:
                st.markdown("#### 🍃 Biological & Eco-Friendly Management")
                st.markdown("These interventions minimize environmental impact and are suitable for organic certified farming:")
                for o in disease_info.get("organic_treatment", []):
                    st.markdown(f"- 🌿 **{o}**")

            with tab_chem:
                st.markdown("#### 🧪 Chemical Control & Targeted Therapeutics")
                st.markdown("Recommended active ingredients and spray schedules for commercial or high-severity outbreaks:")
                for c in disease_info.get("chemical_treatment", []):
                    st.markdown(f"- 💊 {c}")
                st.warning("⚠️ **Safety Notice:** Always follow local pesticide registration labels, observe Pre-Harvest Intervals (PHI), and use proper personal protective equipment (gloves, mask, goggles).")

            with tab_prev:
                st.markdown("#### 🛡️ Long-term Cultural & Preventive Strategies")
                for p in disease_info.get("prevention", []):
                    st.markdown(f"- 🛡️ {p}")

            with tab_chat:
                st.markdown("#### 💬 Ask the AI Agronomist")
                st.markdown("Have specific questions about this crop or disease? Ask for personalized recommendations.")

                # Quick Question Chips
                q_col1, q_col2, q_col3 = st.columns(3)
                user_q = ""
                if q_col1.button("⏱️ How often to spray?"):
                    user_q = "How often should I spray treatment for this disease and what weather should I watch for?"
                if q_col2.button("🍎 Is the fruit safe to eat?"):
                    user_q = "Is the fruit safe to harvest and eat with this disease present?"
                if q_col3.button("🍃 Best organic home remedies?"):
                    user_q = "What are the best organic and natural remedies to stop this disease from spreading?"

                custom_q = st.text_input("Or type your own question:", value=user_q, placeholder="e.g., Can I apply copper spray during flowering?")

                if st.button("Get Agronomist Advice", type="primary") or custom_q:
                    active_query = custom_q if custom_q else "Provide a detailed treatment and prevention plan for this condition."
                    with st.spinner("Consulting Agronomic Knowledge Base..."):
                        consultation_reply = get_ai_agronomic_consultation(
                            crop=disease_info.get("crop", "Crop"),
                            disease=disease_info.get("disease", "Condition"),
                            user_query=active_query,
                            disease_info=disease_info,
                            api_key=api_key_input
                        )
                        st.markdown(consultation_reply)

            # Download Report
            st.divider()
            report_content = format_markdown_report(disease_info, confidence)
            st.download_button(
                label="📥 Download Diagnostic & Treatment Report (.MD)",
                data=report_content,
                file_name=f"CropGuard_Report_{disease_info.get('crop')}_{disease_info.get('disease')}.md",
                mime="text/markdown"
            )

    else:
        st.info("👆 Please upload an image, capture one via camera, or choose a sample from the test gallery above to begin.")


# =========================================================
# PAGE 2: DISEASE DATABASE EXPLORER
# =========================================================
elif nav_selection == "📚 Disease Database (38 Classes)":
    st.markdown("<div class='main-header'>📚 Crop Disease Knowledge Base</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Explore symptoms, causal pathogens, and treatment protocols for all 38 supported crop conditions.</div>", unsafe_allow_html=True)

    col_filter, col_search = st.columns([1, 2])
    all_crops = ["All Crops"] + get_all_crops()

    with col_filter:
        selected_crop_filter = st.selectbox("Filter by Crop:", all_crops)

    with col_search:
        search_query = st.text_input("🔍 Search by Disease, Symptom, or Pathogen:", placeholder="e.g. Blight, Rust, Venturia, concentric rings...")

    # Filter records
    db = load_database()
    matched_records = []

    for key, item in db.items():
        # Crop filter
        if selected_crop_filter != "All Crops" and item.get("crop") != selected_crop_filter:
            continue
        # Search query
        if search_query.strip():
            query_str = search_query.lower()
            corpus = f"{item.get('crop')} {item.get('disease')} {item.get('pathogen')} {item.get('description')} {' '.join(item.get('symptoms', []))}".lower()
            if query_str not in corpus:
                continue
        matched_records.append((key, item))

    st.markdown(f"**Found {len(matched_records)} matching conditions:**")

    for key, record in matched_records:
        is_healthy = record.get("status") == "Healthy"
        icon = "🌿" if is_healthy else "⚠️"
        title = f"{icon} {record.get('crop')} — {record.get('disease')} ({record.get('status')})"

        with st.expander(title, expanded=False):
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(f"**Pathogen:** `{record.get('pathogen')}`")
                st.markdown(f"**Severity:** `{record.get('severity')}`")
                st.markdown(f"**Description:** {record.get('description')}")
                st.markdown("**Key Symptoms:**")
                for s in record.get("symptoms", []):
                    st.markdown(f"- {s}")
            with c2:
                st.markdown("**Organic Treatments:**")
                for o in record.get("organic_treatment", []):
                    st.markdown(f"- 🌿 {o}")
                st.markdown("**Chemical Treatments:**")
                for ch in record.get("chemical_treatment", []):
                    st.markdown(f"- 🧪 {ch}")
                st.markdown(f"**Conducive Weather:** {record.get('favorable_weather')}")


# =========================================================
# PAGE 3: MODEL & ARCHITECTURE
# =========================================================
elif nav_selection == "🧠 Model & Architecture":
    st.markdown("<div class='main-header'>🧠 CropGuard AI Architecture & Deep Learning Specifications</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>System diagram, convolutional neural network layer details, and pipeline components.</div>", unsafe_allow_html=True)

    st.markdown("### 🏗️ High-Level System Architecture")
    st.markdown("""
```text
CropGuard AI
│
├── Frontend / UI
│   └── Streamlit (Modern Web Interface & Interactive Dashboard)
│
├── AI Model
│   └── Pre-trained Convolutional Neural Network (38 Crop Classes)
│
├── Disease Database
│   └── Disease → symptoms → treatment → prevention
│
└── Python Backend
    └── Connects image → model → result → advice
```
    """)

    st.divider()

    st.markdown("### 🔬 Convolutional Neural Network (CNN) Specifications")
    col_arch1, col_arch2 = st.columns(2)

    with col_arch1:
        st.markdown("""
        #### Network Parameters:
        - **Input Resolution:** `128 x 128 x 3` (RGB)
        - **Convolution Blocks:** 5 Sequential Dual-Conv Blocks (10 Conv2D Layers total)
        - **Filter Progression:** `32 → 64 → 128 → 256 → 512`
        - **Activations:** Rectified Linear Unit (`ReLU`)
        - **Pooling:** `MaxPooling2D (2x2, stride 2)` between blocks
        - **Classifier Head:** `Dense(1500 units) + Dropout(0.4)`
        - **Output Layer:** `Dense(38 units, Softmax)`
        """)

    with col_arch2:
        st.markdown("""
        #### Training & Dataset Metadata:
        - **Base Dataset:** New Plant Diseases Dataset (PlantVillage)
        - **Validation Images:** 17,572 labeled crop leaf images
        - **Supported Plant Families:** 14 Major Botanical Families
        - **Crops Included:** Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato
        - **Inference Runtime:** < 50ms per leaf on CPU
        """)

    st.divider()

    st.markdown("### 🔄 End-to-End Pipeline Execution Flow")
    st.markdown("""
    1. **Input Acquisition:** User uploads a leaf image, captures via live camera, or picks a test sample.
    2. **Image Preprocessing:** Tensor normalized, RGB converted, and resized to `(1, 128, 128, 3)`.
    3. **Deep Learning Inference:** CNN processes tensor through 10 convolutional feature extractors and computes 38 softmax probabilities.
    4. **Differential Diagnosis:** Engine extracts top class and ranks secondary candidates to assess uncertainty.
    5. **Database Correlation:** Disease key matches knowledge base to retrieve verified symptoms, pathogens, and remedies.
    6. **Agronomic Advisory Generation:** Offline expert rule engine or Google Gemini API generates customized farmer action plans.
    """)


# =========================================================
# PAGE 4: AGRONOMY BEST PRACTICES
# =========================================================
elif nav_selection == "👨‍🌾 Agronomy Best Practices":
    st.markdown("<div class='main-header'>👨‍🌾 Agricultural Pathology & IPM Field Guide</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Practical guidelines for preventing and managing plant diseases in crops.</div>", unsafe_allow_html=True)

    col_ipm1, col_ipm2 = st.columns(2)

    with col_ipm1:
        st.markdown("""
        ### 🛡️ Integrated Pest & Disease Management (IPM)
        1. **Cultural Prevention (Foundation):**
           - Rotate crops every 2-3 years away from the same botanical family (e.g., Solanaceae: Tomato, Potato, Pepper).
           - Use drip irrigation rather than overhead sprinklers to keep foliage dry.
           - Space plants properly to ensure continuous canopy ventilation and sunlight penetration.
        2. **Sanitation & Hygiene:**
           - Prune lower foliage touching the soil to eliminate soil-splash fungal inoculation.
           - Sterilize pruning shears in 70% alcohol or 10% bleach between plants.
           - Bag and dispose of diseased residues; never compost actively sporulating blight leaves.
        """)

    with col_ipm2:
        st.markdown("""
        ### 🧪 Responsible Fungicide & Spray Management
        1. **Fungicide Resistance Management (FRAC):**
           - Never apply the same single-site systemic fungicide more than twice in succession.
           - Tank-mix or alternate systemic fungicides with multi-site protectants (like copper or mancozeb).
        2. **Spray Timing & Environmental Conditions:**
           - Spray during early morning (6 AM - 9 AM) or late afternoon (after 5 PM) when winds are calm (< 8 km/h).
           - Avoid spraying in full sun during hot days (> 30°C) to prevent foliar phytotoxicity / burn.
           - Wear chemical-resistant gloves, eye protection, and an organic vapor respirator.
        """)

    st.divider()
    st.info("💡 **Farmer Tip:** Early detection is the single most important factor in saving crop yields. Inspect crop leaves weekly, focusing on the oldest lower leaves where early blight, Septoria, and downy mildews typically initiate.")
