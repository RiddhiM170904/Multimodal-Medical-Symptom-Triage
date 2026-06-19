# MY DNA Triage — Multi-Layer AI Clinical Diagnostic System PROJECT

An advanced, multimodal medical decision-support dashboard. The system processes natural language text descriptions, structured vital parameters, and clinical images of skin lesions/wounds simultaneously. It runs a multi-layered classification pipeline **(EfficientNet-B0 + DistilBERT + Random Forest + XGBoost)** and generates agentic triage advice via **Claude 3.5 Haiku**.

The UI follows a high-tech, glassmorphic medical layout with glowing gradients, floating DNA helices, interactive SHAP indicators, preset clinical cases, and configurable simulated/live API modes.

---

## 📂 Repository File Structure

| File | Description |
|------|-------------|
| `index.html` | Main frontend — Hero, Triage Dashboard, Architecture, Colab Guides sections |
| `style.css` | Outfit/Inter typography, deep dark navy aesthetics, neon gradients, pulse animations |
| `app.js` | Patient intake presets, drag-drop upload, SHAP chart renderer, HF Spaces live API connectivity |
| `app_backend.py` | Gradio server backend for Hugging Face Spaces — model loading, inference routing, Claude API, response caching |
| `notebooks/1_Data_Pipeline_and_EDA.ipynb` | Colab: kagglehub download, metadata loading, EDA, HAM10000 Dataset + Dataloader construction |
| `notebooks/2_Model_Training_and_Evaluation.ipynb` | Colab: EfficientNet training, DistilBERT fine-tuning, Random Forest, XGBoost, SHAP |

---

## 🗺️ Project Progress Tracker

| Phase | Task | Status |
|-------|------|--------|
| Phase 1 | Kaggle API setup & license acceptance | ✅ Done |
| Phase 1 | HAM10000 dataset download via `kagglehub` | ✅ Done |
| Phase 1 | Metadata CSV loaded (`HAM10000_metadata.csv`) | ✅ Done |
| Phase 1 | EDA — class distribution plots | ✅ Done |
| Phase 1 | `HAM10000Dataset` + Train/Val DataLoaders | ✅ Done |
| Phase 2 | Train **EfficientNetB0** (Model 1 — Image) | 🔲 Next |
| Phase 2 | Fine-tune **DistilBERT** (Model 2 — Text NLP) | 🔲 Pending |
| Phase 2 | Train **Random Forest** (Model 3 — Symptom Chips) | 🔲 Pending |
| Phase 2 | Train **XGBoost** (Model 4 — Severity Scorer) | 🔲 Pending |
| Phase 2 | SHAP Explainer integration | 🔲 Pending |
| Phase 3 | Hugging Face Spaces deployment | 🔲 Pending |
| Phase 3 | Claude API secret configuration | 🔲 Pending |
| Phase 3 | Frontend live-mode endpoint linking | 🔲 Pending |

---

## 🛠️ Complete Development & Training Step-by-Step

### Prerequisites — Before Opening Colab

1. **Get your Kaggle API Token**
   - Go to [kaggle.com](https://www.kaggle.com) → Profile → **Settings** → **API** → **Create New Token**
   - This downloads `kaggle.json` — keep it ready to upload into Colab

2. **Accept dataset licenses** on Kaggle (one-time, required to avoid 403 errors):
   - [HAM10000 Skin Lesion MNIST](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
   - [Symptom2Disease](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease)
   - [Disease Prediction (132 Symptoms)](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)
   - [Disease Diagnosis + Severity](https://www.kaggle.com/datasets/s3programmer/disease-diagnosis-dataset)

---

### ✅ Phase 1 — Environment + Data Pipeline (COMPLETE)

> **Runtime**: Google Colab → T4 GPU  
> **Notebook**: `notebooks/1_Data_Pipeline_and_EDA.ipynb`

**Key decisions made:**
- Dataset downloaded using `kagglehub.dataset_download()` (more reliable than the legacy `kaggle` CLI)
- Used `HAM10000_metadata.csv` (NOT the flat pixel CSVs `hmnist_28_28_RGB.csv` / `hmnist_8_8_RGB.csv`)
- Fixed class-to-index mapping (`dx_to_label`) that must be consistent across training and backend inference:

```python
dx_to_label = {
    'mel': 0,    # melanoma
    'nv': 1,     # melanocytic nevus
    'bcc': 2,    # basal cell carcinoma
    'akiec': 3,  # actinic keratosis
    'bkl': 4,    # benign keratosis
    'df': 5,     # dermatofibroma
    'vasc': 6    # vascular lesion
}
```

**Class imbalance observed** (major consideration for training):

| Class | Abbr | Count |
|-------|------|-------|
| Melanocytic Nevus | nv | 6,705 |
| Melanoma | mel | 1,113 |
| Benign Keratosis | bkl | 1,099 |
| Basal Cell Carcinoma | bcc | 514 |
| Actinic Keratosis | akiec | 327 |
| Vascular Lesion | vasc | 142 |
| Dermatofibroma | df | 115 |

→ Addressed via **inverse-frequency class weights** in `CrossEntropyLoss` during training.

**Download snippet used:**
```python
import kagglehub
path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
```

---

### 🔲 Phase 2 — Model Training (In Progress)

> **Runtime**: Google Colab → T4 GPU (do NOT restart the session between Phase 1 and Phase 2)  
> **Notebook**: `notebooks/2_Model_Training_and_Evaluation.ipynb`

#### Model 1: EfficientNetB0 — Skin Lesion Image Classifier
- **Dataset**: HAM10000 (10,015 dermoscopy images, 7 classes)
- **Target**: Weighted F1 ≥ **0.82**
- **Method**: Transfer learning from ImageNet weights, Dropout(0.4) head, AdamW + CosineAnnealingLR, 10 epochs
- **Output file**: `best_efficientnet_ham10000.pth` (~18 MB)

#### Model 2: DistilBERT — NLP Symptom Text Classifier
- **Dataset**: Symptom2Disease (24 disease classes)
- **Target**: Weighted F1 ≥ **0.87**
- **Method**: Fine-tune `distilbert-base-uncased` via HuggingFace Trainer, fp16=True, 4 epochs
- **Output folder**: `saved_distilbert_symptom/`

#### Model 3: Random Forest — Structured Symptom Chip Classifier
- **Dataset**: Disease Prediction (132 binary symptom flags)
- **Target**: Accuracy ≥ **0.85**
- **Method**: `RandomForestClassifier(n_estimators=100, max_depth=12)`
- **Output file**: `rf_symptom_binary.pkl`

#### Model 4: XGBoost — Severity Risk Scorer
- **Dataset**: Disease Diagnosis & Severity (Age, Gender, Temp, SpO2, HR, BP → Mild/Moderate/Severe)
- **Target**: Accuracy ≥ **0.80**
- **Method**: `XGBClassifier` + LabelEncoders for gender and severity
- **Output files**: `xgb_severity_scorer.pkl`, `label_encoder_severity.pkl`, `label_encoder_gender.pkl`

**Download all 6 artifacts from Colab after training:**
```python
from google.colab import files
files.download('/content/best_efficientnet_ham10000.pth')
files.download('/content/rf_symptom_binary.pkl')
files.download('/content/xgb_severity_scorer.pkl')
files.download('/content/label_encoder_severity.pkl')
files.download('/content/label_encoder_gender.pkl')
# For the DistilBERT folder, zip it first:
!zip -r /content/saved_distilbert_symptom.zip /content/saved_distilbert_symptom
files.download('/content/saved_distilbert_symptom.zip')
```

Place all files in the **root of this repository**.

---

### 🔲 Phase 3 — Deployment (Free Hosting)

1. **Create a Hugging Face Space**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces) → **Create New Space**
   - SDK: **Gradio** | Tier: **Free CPU Basic** (or T4 Small for GPU)

2. **Upload artifacts to the Space**
   - Rename `app_backend.py` → `app.py` in your Space root
   - Upload all 6 trained model artifacts to the Space root

3. **Configure secrets**
   - Space Settings → **Variables and Secrets** → Add `ANTHROPIC_API_KEY`

4. **Connect the frontend**
   - Copy your Space API URL: `https://yourusername-triage.hf.space/api/predict`
   - In the MYDNA dashboard, click the ⚙️ Settings gear (top right)
   - Switch mode to **Live Hugging Face Spaces Backend**, paste the URL, click **Apply**

---

## 🔧 Recent Code Fixes (app_backend.py)

| Issue | Fix Applied |
|-------|-------------|
| `pretrained=False` deprecated | Updated to `weights=None` (torchvision ≥ 0.13) |
| `torch.load` FutureWarning | Added `weights_only=True` (PyTorch ≥ 2.0) |
| `ham10000_classes` ordering | Now explicitly documented with matching `dx_to_label` index comments |

---

## ⚡ Running the Front-End Locally

```bash
# Option 1 — Node serve
npx serve .

# Option 2 — Python HTTP server
python -m http.server 8000
```

Navigate to `http://localhost:8000` to use the dashboard in **Simulated Mode** (preloaded clinical cases — no backend required).
