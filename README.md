# MY DNA Triage — Multi-Layer AI Clinical Diagnostic System

An advanced, multimodal medical decision-support dashboard. The system processes natural language text descriptions, structured vital parameters, and clinical images of skin lesions/wounds simultaneously. It runs a multi-layered classification pipeline (EfficientNet-B0 + DistilBERT + Random Forest + XGBoost) and generates agentic triage advice via Claude 3.5 Haiku.

The user interface follows the high-tech, glassmorphic medical layout inspired by the [MYDNA Dribbble Design](https://dribbble.com/shots/25565409-MYDNA-Medical-Website-Landing-Page-Design) with glowing gradients, floating DNA helices, presets, interactive SHAP indicators, and customizable simulated/live API switches.

---

## 📂 Repository File Structure

- **`index.html`**: The main frontend landing page and dashboard workspace.
- **`style.css`**: Outfit/Inter typography, deep dark navy aesthetics, neon gradients, pulsing glow visual animations.
- **`app.js`**: Patient intake preset mappings, file dropzone logic, SHAP chart renderer, and Hugging Face endpoint connectivity.
- **`app_backend.py`**: Gradio server backend script to be deployed to Hugging Face Spaces, orchestrating predictions, caching, and Claude API.
- **`notebooks/1_Data_Pipeline_and_EDA.ipynb`**: Colab Notebook for setup, Kaggle API integration, unzipping, class plots, and data splits.
- **`notebooks/2_Model_Training_and_Evaluation.ipynb`**: Colab Notebook for Transfer Learning (EfficientNet), NLP fine-tuning, Random Forest, XGBoost, and SHAP.

---

## 🛠️ Complete Development & Training Step-by-Step

### 1. Kaggle Datasets Needed
Ensure you have access to these Kaggle datasets:
- **Layer 1 (Image)**: [HAM10000 Skin Lesion MNIST](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Layer 2 (Text)**: [Symptom2Disease Dataset](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease) and [Disease Prediction (132 Symptoms)](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)
- **Layer 3 (Severity)**: [Disease Diagnosis + Severity Dataset](https://www.kaggle.com/datasets/s3programmer/disease-diagnosis-dataset)

---

### Phase 1 — Environment + Data Pipeline (Colab Day 1)
1. Open Google Colab and set the runtime type to **T4 GPU** (or V100).
2. Download your `1_Data_Pipeline_and_EDA.ipynb` notebook and upload it to Colab.
3. Download your Kaggle credentials file `kaggle.json` from Kaggle -> Settings -> "Create New API Token".
4. Upload `kaggle.json` to Colab and run the data pipeline cells. This downloads and extracts all three layers of datasets.
5. Review the Exploratory Data Analysis (EDA) charts to understand class distribution imbalances.

---

### Phase 2 — Model Training (Colab Day 2)
1. Upload and run `2_Model_Training_and_Evaluation.ipynb` in Colab.
2. **Train EfficientNetB0** on HAM10000 using PyTorch. 
   - *Target F1/Accuracy*: **82-85%**.
   - *Method*: Fine-tune classification head with Dropout (0.4) and a Cross-Entropy Loss weighted by class inverse frequencies to handle the heavy class imbalance.
3. **Fine-tune DistilBERT** on Symptom2Disease.
   - *Target F1/Accuracy*: **87-91%**.
   - *Method*: Tokenize using `DistilBertTokenizerFast`, set up the Trainer with `CosineAnnealingLR` decay.
4. **Train Random Forest** on the 132-symptom dataset.
   - *Target F1/Accuracy*: **85%+**.
   - *Method*: Fit random forest on binary symptom presence flags and save with `joblib`.
5. **Train XGBoost Severity Scorer** on the Disease Diagnosis + Severity dataset.
   - *Target F1/Accuracy*: **80%+**.
   - *Method*: Train on Patient Age, Gender, Temp, SpO2, Heart Rate, BP, and primary classified disease index.
6. **Download all saved models** from Colab to your machine:
   - `best_efficientnet_ham10000.pth`
   - `saved_distilbert_symptom/`
   - `rf_symptom_binary.pkl`
   - `xgb_severity_scorer.pkl`
   - `label_encoder_severity.pkl`
   - `label_encoder_gender.pkl`

---

### Phase 3 — Deployment (Free Hosting)
1. **Create a Space on Hugging Face**:
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create New Space**.
   - Select **Gradio** as the SDK. Choose the **Free (CPU Basic)** or T4 small space.
2. **Upload Artifacts**:
   - Upload `app_backend.py` (rename it to `app.py` in your Space root).
   - Create a directory called `models/` or place the downloaded training artifacts in the root directory.
3. **Configure Environment Secrets**:
   - Go to your HF Space -> **Settings** -> **Variables and Secrets**.
   - Add a Secret named `ANTHROPIC_API_KEY` and enter your free-tier Claude API key.
4. **Link the space to your MYDNA Frontend**:
   - Once your Space is running, copy the space API URL (e.g. `https://yourusername-triage.hf.space/api/predict`).
   - Open your MYDNA landing page, click the **Settings Gear (Sliders)** in the top right, switch the operation mode to **Live Hugging Face Spaces Backend**, enter the URL, and click Apply.
   - Run tests. The client-side dashboard will now communicate with the Python backend to run actual PyTorch/Transformers code and invoke Claude 3.5 Haiku!

---

## ⚡ Running the Front-End Locally
To launch and view the MYDNA website locally:
1. Double-click `index.html` or run a local server:
   ```bash
   npx serve .
   # or
   python -m http.server 8000
   ```
2. Navigate to `http://localhost:8000` (or the port shown) to experience the responsive UI, presets, and diagnostic results.
