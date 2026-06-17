import os
import joblib
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import gradio as gr
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import anthropic
import hashlib
import json

# ==========================================================================
# 1. Model Definitions & Initialization
# ==========================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image Classifier Definition
# NOTE: Architecture must exactly match the training script in notebooks/2_Model_Training_and_Evaluation.ipynb
class SkinLesionClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super(SkinLesionClassifier, self).__init__()
        # weights=None is the modern equivalent of pretrained=False (torchvision >= 0.13)
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)

# Load Tokenizers and Classifiers
print("Loading clinical models...")

# In a live HF Space, ensure these files are uploaded in the repository:
# - best_efficientnet_ham10000.pth
# - saved_distilbert_symptom/
# - rf_symptom_binary.pkl
# - xgb_severity_scorer.pkl
# - label_encoder_severity.pkl
# - label_encoder_gender.pkl

# Dummy placeholders if models are not present (so space boots up correctly)
MODELS_READY = False
try:
    # Load Image Classifier
    image_model = SkinLesionClassifier(num_classes=7)
    if os.path.exists('best_efficientnet_ham10000.pth'):
        # weights_only=True suppresses FutureWarning on PyTorch >= 2.0
        image_model.load_state_dict(
            torch.load('best_efficientnet_ham10000.pth', map_location=device, weights_only=True)
        )
    image_model.to(device)
    image_model.eval()

    # Load DistilBERT Classifier
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    nlp_model = DistilBertForSequenceClassification.from_pretrained('saved_distilbert_symptom')
    nlp_model.to(device)
    nlp_model.eval()

    # Load Random Forest & XGBoost Models
    rf_model = joblib.load('rf_symptom_binary.pkl')
    xgb_severity = joblib.load('xgb_severity_scorer.pkl')
    le_severity = joblib.load('label_encoder_severity.pkl')
    le_gender = joblib.load('label_encoder_gender.pkl')
    
    MODELS_READY = True
    print("All models loaded successfully.")
except Exception as e:
    print(f"Warning: Models failed to load ({e}). Running in Mock Fallback Mode.")

# Image pre-processing transformations
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Clinical labels maps
# Class order MUST match dx_to_label used during training:
# dx_to_label = {'mel':0, 'nv':1, 'bcc':2, 'akiec':3, 'bkl':4, 'df':5, 'vasc':6}
ham10000_classes = [
    'melanoma',            # 0 — mel
    'melanocytic nevus',   # 1 — nv
    'basal cell carcinoma',# 2 — bcc
    'actinic keratosis',   # 3 — akiec
    'benign keratosis',    # 4 — bkl
    'dermatofibroma',      # 5 — df
    'vascular lesion'      # 6 — vasc
]
symptom2disease_classes = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction', 'Peptic ulcer disease', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension', 'Migraine', 'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E']

# Cache for Claude API calls (to respect free tier limits)
response_cache = {}

# ==========================================================================
#  2. Helper Logic & API Inferences
# ==========================================================================

def get_cache_key(*args):
    """Generates a unique md5 hash key for caching."""
    serialized = json.dumps(list(args), sort_keys=True)
    return hashlib.md5(serialized.encode('utf-8')).hexdigest()

def predict_lesion(img_path):
    """Classify skin image using EfficientNetB0."""
    if not MODELS_READY:
        return [{"name": "Benign Keratosis", "pct": 76}, {"name": "Dermatofibroma", "pct": 15}, {"name": "Melanoma", "pct": 9}]
        
    try:
        img = Image.open(img_path).convert('RGB')
        tensor = image_transforms(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = image_model(tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()
            
        indices = np.argsort(probs)[::-1][:3]
        return [{"name": ham10000_classes[idx], "pct": int(probs[idx] * 100)} for idx in indices]
    except Exception as e:
        print(f"Image inference error: {e}")
        return [{"name": "Error classifying image", "pct": 0}]

def predict_symptom_text(text):
    """Classify symptom description using DistilBERT."""
    if not MODELS_READY:
        return [{"name": "Contact Dermatitis", "pct": 88}, {"name": "Eczema", "pct": 8}, {"name": "Psoriasis", "pct": 4}]
        
    try:
        inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = nlp_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits[0], dim=0).cpu().numpy()
            
        indices = np.argsort(probs)[::-1][:3]
        return [{"name": symptom2disease_classes[idx], "pct": int(probs[idx] * 100)} for idx in indices]
    except Exception as e:
        print(f"Text inference error: {e}")
        return [{"name": "Error classifying text", "pct": 0}]

def get_xgb_severity(age, gender, temp, spo2, hr, bp, text_dx_code):
    """Classify case severity using XGBoost."""
    if not MODELS_READY:
        # Fallback heuristic
        if spo2 < 92 or temp > 39.0:
            return "severe", 85, "Emergency Bypass", "Red"
        elif temp > 38.0 or hr > 95:
            return "moderate", 45, "Standard Triage", "Yellow"
        return "mild", 15, "Standard Triage", "Green"
        
    try:
        # Encode gender
        gender_code = le_gender.transform([gender])[0]
        
        # Features array
        features = np.array([[age, gender_code, temp, spo2, hr, bp, text_dx_code]])
        prob = xgb_severity.predict_proba(features)[0]
        pred = xgb_severity.predict(features)[0]
        
        severity_label = le_severity.inverse_transform([pred])[0].lower() # 'mild', 'moderate', 'severe'
        risk_score = int(np.max(prob) * 100)
        
        pathway = "Emergency Bypass" if severity_label == "severe" else "Standard Triage"
        gate = "Red" if severity_label == "severe" else ("Yellow" if severity_label == "moderate" else "Green")
        
        return severity_label, risk_score, pathway, gate
    except Exception as e:
        print(f"Severity scoring error: {e}")
        return "mild", 10, "Standard Triage", "Green"

def query_claude(prompt, cache_key):
    """Fetch advice from Claude 3.5 Haiku (supporting key checks and caching)."""
    # Check cache
    if cache_key in response_cache:
        return response_cache[cache_key], True
        
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "<h4>Primary Assessment</h4><p>Claude API key not configured. Running in offline/simulation advice mode.</p>", False
        
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            temperature=0.3,
            system="You are a clinical decision-support assistant. Provide clear, structured medical triage recommendations. Strictly include safety boundaries and clarify that this is not a final diagnosis.",
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = message.content[0].text
        # Cache the response
        response_cache[cache_key] = response_text
        return response_text, False
    except Exception as e:
        return f"<h4>Error contacting Claude AI API</h4><p>{e}</p>", False

# ==========================================================================
# 3. Main Triage Controller Endpoint
# ==========================================================================

def execute_triage(text_symptoms, image, age, gender, temp, spo2, hr, bp, active_chips):
    # Determine cache key
    cache_key = get_cache_key(text_symptoms, age, temp, spo2, hr, bp, active_chips)
    
    # Run Layer 1 & 2 Classifiers
    text_probs = predict_symptom_text(text_symptoms)
    image_probs = predict_lesion(image) if image else []
    
    # Translate text diagnosis name to numeric index for XGBoost
    primary_text_dx = text_probs[0]["name"]
    text_dx_code = symptom2disease_classes.index(primary_text_dx) if primary_text_dx in symptom2disease_classes else 0
    
    # Run Layer 3 Classifier
    severity, risk, pathway, gate = get_xgb_severity(age, gender, temp, spo2, hr, bp, text_dx_code)
    
    # Calculate SHAP Explainer representation (Mocked based on feature deviations)
    shap_features = []
    # Base indicators of SHAP values
    temp_diff = temp - 36.8
    spo2_diff = 98 - spo2
    hr_diff = hr - 72
    
    shap_features.append({"feature": "SpO2 Saturation", "val": float(spo2_diff * 0.6), "dir": "positive" if spo2_diff > 0 else "negative"})
    shap_features.append({"feature": "Body Temp", "val": float(temp_diff * 1.5), "dir": "positive" if temp_diff > 0 else "negative"})
    shap_features.append({"feature": "Heart Rate", "val": float(hr_diff * 0.1), "dir": "positive" if hr_diff > 0 else "negative"})
    shap_features.append({"feature": "Age Factor", "val": float((age - 30) * 0.02), "dir": "positive" if age > 30 else "negative"})
    
    # Gating & LLM Layer
    if severity == "severe" or spo2 < 90:
        # Emergency Gate - Bypass Claude to save latency/limits, return emergency template
        claude_advice = f"""<div class="emergency-notice">
            <h4><i class="fa-solid fa-circle-exclamation"></i> CRITICAL MEDICAL EMERGENCY</h4>
            <p>The patient profile suggests acute risk. Oxygen saturation is low ({spo2}%) or vitals indicate sepsis/shock factors. Immediate hospital intervention is required.</p>
        </div>
        <h4>Immediate Actions</h4>
        <ul>
            <li><strong>Emergency Services:</strong> Dial 911 or your local emergency line immediately.</li>
            <li><strong>Rest:</strong> Sit upright in a comfortable position if breathing is difficult; avoid any physical exertion.</li>
            <li><strong>Monitoring:</strong> Continually check consciousness levels and pulse rates.</li>
        </ul>"""
        is_cached = False
    else:
        # Prompt construction
        prompt = f"""
        Provide triage recommendations for the following patient:
        - Age: {age}, Gender: {gender}
        - Vitals: Temperature {temp}°C, SpO2 {spo2}%, Heart Rate {hr} bpm, Blood Pressure (Systolic) {bp} mmHg
        - Primary Symptom Description: "{text_symptoms}"
        - ML Classified Indication: {primary_text_dx} (Probability: {text_probs[0]["pct"]}%)
        - Risk Severity Score: {risk}% (Triage Class: {severity.upper()})
        
        Generate a structured clinical report in HTML format. Write section headers using <h4>, lists using <ul>, and emphasize crucial parameters. Do not include markdown codeblocks (e.g. ```html). Format it with the following sections:
        - <h4>Primary Assessment</h4>: Analyze the case and probable causes.
        - <h4>Suggested Next Steps</h4>: Self-care, clinical routes, and specialist referral type.
        - <h4>Safety Precautions</h4>: Warnings on worsening signs, when to seek immediate emergency care.
        """
        claude_advice, is_cached = query_claude(prompt, cache_key)
        
    return [
        severity,
        risk,
        pathway,
        gate,
        text_probs,
        image_probs,
        shap_features,
        claude_advice
    ]

# ==========================================================================
# 4. Gradio Interface Construction
# ==========================================================================

gr_interface = gr.Interface(
    fn=execute_triage,
    inputs=[
        gr.Textbox(label="Symptom Text"),
        gr.Image(type="filepath", label="Lesion Image"),
        gr.Number(label="Age", value=28),
        gr.Dropdown(choices=["Male", "Female", "Other"], label="Gender", value="Male"),
        gr.Number(label="Temperature", value=36.8),
        gr.Number(label="SpO2", value=98),
        gr.Number(label="Heart Rate", value=72),
        gr.Number(label="BP Systolic", value=120),
        gr.Textbox(label="Active Chips (Comma separated)")
    ],
    outputs="json",
    title="MYDNA Symptom Triage Engine API Endpoint",
    description="Backend API wrapper supporting multimodal input triage routing and Gradio requests."
)

if __name__ == "__main__":
    gr_interface.launch()
