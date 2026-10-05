"""
CropGuard AI - Core Pipeline Engine
Connects: image -> model -> result -> advice
"""

from typing import Dict, Any, Optional, Union
from PIL import Image
from backend.model import predict_disease
from backend.database import get_disease_info, format_markdown_report
from backend.advisory import get_ai_agronomic_consultation


class CropGuardEngine:
    """End-to-end diagnostic and advisory pipeline orchestrator."""

    def __init__(self):
        pass

    def run_pipeline(
        self,
        image_input: Union[str, bytes, Image.Image, Any],
        user_query: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute the full end-to-end diagnostic workflow:
        1. Process Image
        2. Run AI Model Inference
        3. Retrieve Disease Database Record
        4. Generate Expert Actionable Advisory
        """
        # Step 1 & 2: Inference
        prediction_res = predict_disease(image_input)
        primary_class = prediction_res["class_name"]
        confidence = prediction_res["confidence"]

        # Step 3: Disease Database Retrieval
        disease_record = get_disease_info(primary_class)

        # Step 4: Advisory & Guidance
        default_query = user_query or f"What are the immediate treatment steps and spray precautions for {disease_record.get('disease')}?"
        ai_advice = get_ai_agronomic_consultation(
            crop=disease_record.get("crop", "Crop"),
            disease=disease_record.get("disease", "Condition"),
            user_query=default_query,
            disease_info=disease_record,
            api_key=api_key
        )

        # Markdown Exportable Summary
        report_md = format_markdown_report(disease_record, confidence)

        return {
            "prediction": prediction_res,
            "disease_record": disease_record,
            "ai_consultation": ai_advice,
            "report_markdown": report_md
        }


# Global singleton engine instance
engine = CropGuardEngine()
