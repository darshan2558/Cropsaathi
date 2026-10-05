"""
CropGuard AI - Advisory & Agronomist Engine
Combines structured database advice with dynamic AI consultation (Gemini or offline Agronomist).
"""

import os
from typing import Dict, Any, Optional

try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


def configure_gemini(api_key: Optional[str] = None) -> bool:
    """Configure Gemini API using provided key or environment variable."""
    if not _GENAI_AVAILABLE:
        return False

    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        return False

    try:
        genai.configure(api_key=key)
        return True
    except Exception:
        return False


def get_ai_agronomic_consultation(
    crop: str,
    disease: str,
    user_query: str,
    disease_info: Dict[str, Any],
    api_key: Optional[str] = None
) -> str:
    """Provide intelligent expert agronomic guidance.
    Uses Gemini when configured; otherwise falls back to the built-in Expert Advisory System.
    """
    # Try Gemini first if key available
    if _GENAI_AVAILABLE:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if key:
            try:
                genai.configure(api_key=key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = f"""
You are an expert plant pathologist and agronomist for CropGuard AI.
A farmer has scanned a {crop} plant, diagnosed with: {disease}.
Pathogen: {disease_info.get('pathogen', 'N/A')}
Status: {disease_info.get('status', 'N/A')}
Symptoms: {', '.join(disease_info.get('symptoms', []))}
Organic Options: {', '.join(disease_info.get('organic_treatment', []))}
Chemical Options: {', '.join(disease_info.get('chemical_treatment', []))}
Prevention: {', '.join(disease_info.get('prevention', []))}

The farmer asks: "{user_query}"

Provide practical, highly actionable, safe, and professional agronomic advice.
Use clear headings, bullet points, and safety warnings where chemicals are involved.
Keep the tone encouraging, respectful, and farmer-friendly.
"""
                response = model.generate_content(prompt)
                if response and response.text:
                    return response.text
            except Exception as e:
                # Fall back seamlessly
                pass

    # Built-in offline Expert Agronomist Engine
    return _generate_expert_offline_advisory(crop, disease, user_query, disease_info)


def _generate_expert_offline_advisory(
    crop: str,
    disease: str,
    user_query: str,
    info: Dict[str, Any]
) -> str:
    """Offline domain-expert agronomist responder that generates tailored answers."""
    q = user_query.lower()
    is_healthy = info.get("status") == "Healthy"

    if is_healthy:
        return f"""### 🌿 Agronomist Assessment: {crop} is Healthy!

Great news! Your **{crop}** shows no active symptoms of infectious pathology.

**Maintenance Recommendations:**
- **Watering:** Provide steady, deep watering at root level (avoiding prolonged foliage wetness).
- **Nutrition:** Ensure balanced N-P-K fertilization suited for {crop}'s current growth stage.
- **Monitoring:** Scout leaf undersides weekly for early signs of mites, aphids, or fungal spores.
- **Canopy:** Maintain proper pruning for sunlight and air passage.

*Feel free to ask questions about preventative nutrition or optimal growing conditions.*"""

    # For diseased crops, answer specific common farmer questions
    organics = "\n".join([f"  • {item}" for item in info.get("organic_treatment", [])])
    chemicals = "\n".join([f"  • {item}" for item in info.get("chemical_treatment", [])])
    prevention = "\n".join([f"  • {item}" for item in info.get("prevention", [])])
    symptoms = "\n".join([f"  • {item}" for item in info.get("symptoms", [])])
    weather = info.get("favorable_weather", "Variable weather")

    if any(k in q for k in ["spray", "often", "when", "frequency", "schedule", "how much"]):
        return f"""### ⏱️ Recommended Application & Spray Schedule for {disease}

For **{crop}** affected by **{disease}** ({info.get('pathogen', 'Pathogen')}):

1. **Immediate Initial Action (Day 1 - 3):**
   - Mechanically remove and safely dispose of heavily infected leaves or shoots to reduce spore load.
   - Do NOT compost infected plant material.
   
2. **First Treatment Application:**
   - Apply protective spray ({info.get('organic_treatment', ['copper or sulfur'])[0]}) in early morning or late afternoon (avoid spraying under intense midday sun).
   
3. **Re-application Intervals:**
   - **During Wet/Humid Periods:** Repeat applications every 5 to 7 days.
   - **During Dry Periods:** Repeat every 10 to 14 days until new foliage emerges symptom-free.
   - If rainfall exceeds 25 mm within 24 hours of application, re-spray promptly.

4. **Safety & Rotation Precautions:**
   - Always rotate between different chemical classes (FRAC codes) to avoid pathogen resistance.
   - Wear protective gloves, goggles, and respiratory masks when handling spray concentrates."""

    if any(k in q for k in ["eat", "edible", "safe to eat", "consume", "toxic", "poison"]):
        return f"""### 🍎 Food Safety & Consumption Advisory for {crop}

Regarding **{crop}** with **{disease}**:

- **Foliage vs Fruit:** The detected pathogen primarily targets leaves/stems. If fruit is unblemished, firm, and fully mature, it is generally safe to consume after thorough washing with clean water.
- **Infected Fruit:** If the disease has visibly scarred, rotted, or caused lesions on the fruit itself, discard or cull the damaged fruit. Fungal and bacterial lesions can introduce secondary mold contaminants.
- **Chemical Pre-Harvest Interval (PHI):** If you have recently sprayed chemical fungicides or bactericides, **strictly respect the product's Pre-Harvest Interval (PHI)** printed on the pesticide label before harvesting."""

    if any(k in q for k in ["organic", "natural", "home remedy", "bio", "neem"]):
        return f"""### 🍃 Organic & Natural Treatment Protocol for {disease}

Here is the targeted organic management strategy for **{crop}**:

{organics}

**Key Cultural Hygiene Steps:**
- Ensure high sanitation: Sterilize pruning shears with 70% rubbing alcohol between each plant.
- Apply clean organic mulch (straw or wood chips) to prevent rain-splash transmission of soil-borne spores.
- Increase plant-to-plant spacing or prune internal foliage to maximize airflow and keep leaves dry."""

    if any(k in q for k in ["chemical", "fungicide", "medicine", "drug", "cure"]):
        return f"""### 🧪 Chemical Control & Therapeutic Strategy for {disease}

Recommended chemical options for severe or commercial outbreaks of **{disease}**:

{chemicals}

**Application Best Practices:**
- Spray during calm weather (wind speed < 8 km/h) to prevent spray drift.
- Ensure thorough coverage of both upper and lower leaf surfaces.
- Strictly adhere to Worker Protection Standards (WPS) and Restricted Entry Intervals (REI)."""

    # Comprehensive Default Response
    return f"""### 👨‍🌾 Agronomic Consultation: {disease} on {crop}

**Pathogen Profile:** {info.get('pathogen', 'Fungal / Bacterial')}  
**Weather Risk Trigger:** {weather}  

#### 🔍 Symptoms to Monitor:
{symptoms}

#### 🍃 Organic & Cultural Solutions:
{organics}

#### 🧪 Chemical Control Measures:
{chemicals}

#### 🛡️ Long-term Field Prevention:
{prevention}

*Tip: You can connect your Google Gemini API key in the sidebar for personalized conversational responses!*"""
