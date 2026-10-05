# 🌿 CropGuard AI — Intelligent Crop Disease Diagnosis & Agronomic Advisory System

CropGuard AI is an end-to-end deep learning and agronomic expert system designed to identify crop foliar diseases from leaf photographs, quantify prediction confidence, and provide actionable organic, chemical, and preventive treatment plans.

---

## 🏗️ System Architecture

CropGuard AI is built strictly according to the 4-tier modular architecture:

```text
CropGuard AI
│
├── Frontend / UI
│   └── Streamlit (Modern interactive dashboard, camera input, sample gallery)
│
├── AI Model
│   └── Pre-trained Deep CNN (10-layer Convolutional Neural Network across 38 crop classes)
│
├── Disease Database
│   └── Disease → symptoms → treatment → prevention (Rich JSON knowledge base)
│
└── Python Backend
    └── Connects image → model → result → advice (Pipeline orchestrator)
```

---

## 🌟 Key Features

1. **Multi-Modal Image Input:**
   - 📁 **File Upload:** Upload leaf photos directly (JPG, JPEG, PNG).
   - 📸 **Live Camera:** Real-time capture via webcam or mobile camera.
   - 🖼️ **Test Sample Gallery:** Test pre-loaded leaf images across diseases (Apple Scab, Cedar Apple Rust, Corn Rust, Potato Blight, Tomato Curl Virus, etc.).

2. **Accurate Deep Learning Inference:**
   - Powered by a 10-layer Convolutional Neural Network trained on the PlantVillage dataset (17,500+ validation images).
   - Classifies across **38 distinct categories** covering 14 crop species.
   - Displays prediction confidence percentage and top runner-up candidate predictions for differential diagnosis.

3. **Exhaustive Agronomic Disease Knowledge Base:**
   - Detailed biological profiles, causal pathogens, and weather conducive to outbreak.
   - **Organic & Cultural Controls:** Biological controls, eco-friendly fungicides, and pruning.
   - **Chemical Treatments:** Specific registered active ingredients, spray timing, and safety warnings.
   - **Long-Term Field Prevention:** Resistant varieties, irrigation advice, and crop rotation.

4. **AI Agronomist Consultation:**
   - Interactive Q&A for farmers (spray schedules, food safety, organic alternatives).
   - Works **offline** out-of-the-box with an internal expert rule engine.
   - Optionally connects to **Google Gemini API** for live conversational AI advice.

5. **Exportable Diagnostic Reports:**
   - Download complete diagnosis summaries in formatted Markdown for farm record-keeping.

---

## 📂 Project Structure

```text
crop project/
├── .streamlit/
│   └── config.toml                  # Streamlit custom green theme & server configuration
├── backend/
│   ├── __init__.py
│   ├── advisory.py                  # AI Agronomist consultation engine (Gemini + offline expert)
│   ├── database.py                  # Knowledge base accessor, search, and Markdown generator
│   ├── engine.py                    # Pipeline coordinator: image -> model -> result -> advice
│   └── model.py                     # Deep learning inference, tensor preprocessing, and caching
├── data/
│   └── disease_database.json        # Comprehensive database for all 38 crop disease classes
├── models/
│   └── trained_plant_disease_model.h5  # Trained 10-layer CNN model weights (94 MB)
├── samples/                         # Curated sample leaf images for instant testing
├── app.py                           # Main Streamlit web application
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation
```

---

## 🚀 Quick Start Guide

### 1. Run the Streamlit Application

Execute the following command in your terminal:

```powershell
& "C:\Users\darsh\anaconda3\envs\tensorflow_env\python.exe" -m streamlit run app.py
```

Or if your conda environment is activated:

```bash
streamlit run app.py
```

### 2. Open in Browser

The web application will automatically open in your default browser at:
`http://localhost:8501`

---

## 🌾 Supported Crops & Diseases (38 Classes)

| Crop | Supported Conditions |
|---|---|
| **Apple** | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| **Blueberry** | Healthy |
| **Cherry** | Powdery Mildew, Healthy |
| **Corn (Maize)** | Cercospora Leaf Spot (Gray Leaf Spot), Common Rust, Northern Leaf Blight, Healthy |
| **Grape** | Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis), Healthy |
| **Orange** | Huanglongbing (Citrus Greening) |
| **Peach** | Bacterial Spot, Healthy |
| **Pepper (Bell)** | Bacterial Spot, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Raspberry** | Healthy |
| **Soybean** | Healthy |
| **Squash** | Powdery Mildew |
| **Strawberry** | Leaf Scorch, Healthy |
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Tomato Yellow Leaf Curl Virus, Tomato Mosaic Virus, Healthy |

---

## 🔒 Optional: Configure Gemini API Key

To enable Google Gemini conversational advice:
1. Open the **⚙️ AI Advisory Settings** expander in the sidebar.
2. Paste your Gemini API Key.
*(Note: If no key is entered, CropGuard AI seamlessly uses its built-in offline expert agronomic engine).*
