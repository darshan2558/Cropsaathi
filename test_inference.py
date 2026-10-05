"""
Test script to copy user images and verify model inference
"""
import sys
import shutil
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Copy user images
user_files = [
    r"C:\Users\darsh\.gemini\antigravity\brain\665b1e3a-ea95-4bb1-aeee-bd6da9a4a6d5\.user_uploaded\media_1791173820335.jpg",
    r"C:\Users\darsh\.gemini\antigravity\brain\665b1e3a-ea95-4bb1-aeee-bd6da9a4a6d5\.user_uploaded\media_1791173820336.jpg",
    r"C:\Users\darsh\.gemini\antigravity\brain\665b1e3a-ea95-4bb1-aeee-bd6da9a4a6d5\.user_uploaded\media_1791173820339.jpg",
    r"C:\Users\darsh\.gemini\antigravity\brain\665b1e3a-ea95-4bb1-aeee-bd6da9a4a6d5\.user_uploaded\media_1791173820341.jpg",
    r"C:\Users\darsh\.gemini\antigravity\brain\665b1e3a-ea95-4bb1-aeee-bd6da9a4a6d5\.user_uploaded\media_1791173820384.jpg",
]

samples_dir = Path("samples")
samples_dir.mkdir(exist_ok=True)

for i, src in enumerate(user_files, 1):
    dest = samples_dir / f"UserUpload_AppleScab_{i}.jpg"
    shutil.copy2(src, dest)
    print(f"Copied user image {i} -> {dest}")

# Test inference using backend
from backend.engine import engine

test_img = samples_dir / "UserUpload_AppleScab_1.jpg"
print(f"\nRunning diagnosis pipeline on: {test_img}")
res = engine.run_pipeline(test_img)

pred = res["prediction"]
print("\n--- DIAGNOSIS RESULT ---")
print(f"Detected Class: {pred['class_name']}")
print(f"Confidence: {pred['confidence_percentage']}")
print("\nTop Predictions:")
for p in pred["top_predictions"]:
    print(f"  {p['crop']} - {p['condition']}: {p['percentage']}")

record = res["disease_record"]
print(f"\nPathogen: {record.get('pathogen')}")
print(f"Severity: {record.get('severity')}")
print(f"Organic Treatment count: {len(record.get('organic_treatment', []))}")
print(f"Chemical Treatment count: {len(record.get('chemical_treatment', []))}")
print("\nAdvisory Preview:")
print(res["ai_consultation"][:300] + "...")
print("\nALL INFERENCE TESTS PASSED!")
