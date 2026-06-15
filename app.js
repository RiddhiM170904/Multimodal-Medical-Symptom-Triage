// ==========================================================================
// MYDNA Triage Dashboard Controller
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const triageForm = document.getElementById('triage-form');
    const uploadZone = document.getElementById('upload-zone');
    const imageInput = document.getElementById('image-input');
    const dropzoneDefault = document.getElementById('dropzone-default');
    const dropzonePreview = document.getElementById('dropzone-preview');
    const imagePreview = document.getElementById('image-preview');
    const removeImageBtn = document.getElementById('remove-image-btn');
    
    const openConfigBtn = document.getElementById('open-config-btn');
    const closeConfigBtn = document.getElementById('close-config-btn');
    const saveConfigBtn = document.getElementById('save-config-btn');
    const resetConfigBtn = document.getElementById('reset-config-btn');
    const configModal = document.getElementById('config-modal');
    const backendModeSelect = document.getElementById('backend-mode');
    const hfConfigFields = document.getElementById('hf-config-fields');
    const hfEndpointInput = document.getElementById('hf-endpoint');
    const claudeKeyInput = document.getElementById('claude-key');
    
    const scanningOverlay = document.getElementById('scanning-overlay');
    const scanStatusText = document.getElementById('scan-status-text');
    const resultPlaceholder = document.getElementById('result-placeholder');
    const resultContent = document.getElementById('result-content');
    
    // Result Elements
    const severityBanner = document.getElementById('severity-banner');
    const severityStatusText = document.getElementById('severity-status-text');
    const xgboostRiskScore = document.getElementById('xgboost-risk-score');
    const clinicalPathway = document.getElementById('clinical-pathway');
    const urgencyGate = document.getElementById('urgency-gate');
    const textPrimaryDisease = document.getElementById('text-primary-disease');
    const textProbBar = document.getElementById('text-prob-bar');
    const textPredictionsList = document.getElementById('text-predictions-list');
    const imagePrimaryDisease = document.getElementById('image-primary-disease');
    const imageProbBar = document.getElementById('image-prob-bar');
    const imagePredictionsList = document.getElementById('image-predictions-list');
    const imageClassifierGroup = document.getElementById('image-classifier-prob-group');
    const shapChartContainer = document.getElementById('shap-chart-container');
    const claudeTextResponse = document.getElementById('claude-text-response');
    const claudeCacheIndicator = document.getElementById('claude-cache-indicator');

    // System Settings State
    let systemConfig = {
        mode: 'simulated',
        hfEndpoint: '',
        claudeKey: ''
    };

    // Preset Cases Library
    const casesPreset = {
        rash: {
            age: 28,
            gender: 'Female',
            temp: 36.8,
            spo2: 98,
            hr: 72,
            bp: 120,
            symptoms: ['itching', 'skin_rash'],
            text: 'I noticed an itchy red rash on my inner arm three days ago, which has started blistering slightly and feels warm. I think it might be contact dermatitis from a new laundry detergent.',
            hasImage: true,
            imageMockUrl: 'https://images.unsplash.com/photo-1606101266946-b258380e9227?auto=format&fit=crop&w=400&q=80',
            disease: 'Contact Dermatitis',
            severity: 'mild',
            risk: 18,
            pathway: 'Standard Triage',
            gate: 'Green',
            textProbs: [
                { name: 'Contact Dermatitis', pct: 88 },
                { name: 'Eczema', pct: 8 },
                { name: 'Psoriasis', pct: 4 }
            ],
            imageProbs: [
                { name: 'Benign Keratosis', pct: 76 },
                { name: 'Dermatofibroma', pct: 15 },
                { name: 'Melanoma', pct: 9 }
            ],
            shap: [
                { feature: 'Skin Rash Flag', val: 2.2, dir: 'positive' },
                { feature: 'Itching Flag', val: 1.5, dir: 'positive' },
                { feature: 'Patient Age', val: -0.4, dir: 'negative' },
                { feature: 'Body Temp', val: -0.2, dir: 'negative' }
            ],
            claudeAdvice: `<h4>Primary Assessment</h4>
            <p>Clinical indications point towards acute <strong>Contact Dermatitis</strong>, likely triggered by localized exposure to a topical irritant or allergen. Mild severity is supported by stable vital signs and absence of systemic features.</p>
            
            <h4>Suggested Next Steps</h4>
            <ul>
                <li><strong>Local relief:</strong> Apply a cool, damp compress to the area for 10-15 minutes to reduce inflammation.</li>
                <li><strong>Topical treatment:</strong> Consider over-the-counter 1% hydrocortisone cream or calamine lotion to manage pruritus.</li>
                <li><strong>Allergen avoidance:</strong> Identify and eliminate recent new products (soaps, laundry detergents, cosmetics).</li>
                <li><strong>Skin barrier protection:</strong> Apply bland, unscented moisturizers or petroleum jelly.</li>
            </ul>
            
            <h4>Safety Precautions</h4>
            <p>Do not scratch the affected area to prevent secondary bacterial infections. Seek medical evaluation if the rash spreads to the face, eyes, or genitals, or if signs of infection (pus, increasing warmth, red streaks) develop.</p>`
        },
        emergency: {
            age: 35,
            gender: 'Male',
            temp: 37.2,
            spo2: 88,
            hr: 115,
            bp: 90,
            symptoms: ['skin_rash', 'breathlessness', 'fatigue'],
            text: 'A sudden hives-like rash is spreading rapidly across my neck and chest. I am having a very hard time breathing, my throat feels tight and swollen, and I am feeling extremely dizzy and weak.',
            hasImage: false,
            disease: 'Anaphylaxis',
            severity: 'severe',
            risk: 89,
            pathway: 'Emergency Bypass',
            gate: 'Red',
            textProbs: [
                { name: 'Anaphylaxis', pct: 94 },
                { name: 'Severe Urticaria', pct: 4 },
                { name: 'Asthma Attack', pct: 2 }
            ],
            shap: [
                { feature: 'SpO2 Saturation', val: 4.8, dir: 'positive' },
                { feature: 'Shortness of Breath', val: 3.2, dir: 'positive' },
                { feature: 'Heart Rate', val: 2.1, dir: 'positive' },
                { feature: 'Blood Pressure', val: 1.8, dir: 'positive' }
            ],
            claudeAdvice: `<div class="emergency-notice">
                <h4><i class="fa-solid fa-circle-exclamation"></i> CRITICAL MEDICAL EMERGENCY</h4>
                <p>The patient profile suggests acute <strong>Anaphylaxis</strong> (severe systemic allergic reaction) characterized by severe respiratory compromise (SpO2 88%), tachycardia (115 bpm), and hypotension (90 mmHg Systolic).</p>
            </div>
            
            <h4>Immediate Urgent Actions</h4>
            <ul>
                <li><strong>Call Emergency Services:</strong> Call 911 or your local emergency number immediately.</li>
                <li><strong>Epinephrine:</strong> If an epinephrine auto-injector (EpiPen) is available, administer it immediately into the outer thigh.</li>
                <li><strong>Positioning:</strong> Lie flat on your back with feet elevated to support blood pressure. If breathing is difficult, sit up slightly but avoid sudden standing.</li>
                <li><strong>Airway:</strong> Do not ingest oral medications, food, or fluids due to high choking risk.</li>
            </ul>
            
            <h4>Clinical Note</h4>
            <p>Emergency medical services must be contacted regardless of whether epinephrine is administered, as a biphasic reaction (reoccurrence of symptoms) can occur within hours.</p>`
        },
        psoriasis: {
            age: 42,
            gender: 'Other',
            temp: 37.8,
            spo2: 97,
            hr: 85,
            bp: 130,
            symptoms: ['skin_rash', 'joint_pain', 'itching'],
            text: 'I have thick, red, scaly patches with silvery scales on my elbows and knees. It has been persistent for months, and now the joints in my fingers are starting to feel stiff, swollen, and painful.',
            hasImage: true,
            imageMockUrl: 'https://images.unsplash.com/photo-1581594693702-fbdc51b2763b?auto=format&fit=crop&w=400&q=80',
            disease: 'Psoriasis',
            severity: 'moderate',
            risk: 48,
            pathway: 'Standard Triage',
            gate: 'Yellow',
            textProbs: [
                { name: 'Psoriasis', pct: 81 },
                { name: 'Rheumatoid Arthritis', pct: 12 },
                { name: 'Dermatitis', pct: 7 }
            ],
            imageProbs: [
                { name: 'Psoriasis / Eczema', pct: 68 },
                { name: 'Melanocytic Nevus', pct: 20 },
                { name: 'Basal Cell Carcinoma', pct: 12 }
            ],
            shap: [
                { feature: 'Joint Pain Flag', val: 2.8, dir: 'positive' },
                { feature: 'Skin Rash Flag', val: 1.9, dir: 'positive' },
                { feature: 'Body Temp', val: 0.6, dir: 'positive' },
                { feature: 'Systolic BP', val: 0.3, dir: 'positive' }
            ],
            claudeAdvice: `<h4>Primary Assessment</h4>
            <p>Symptoms describe classic plaque <strong>Psoriasis</strong> with arthritic involvement (Joint Pain / Stiffness), indicating a progression towards <strong>Psoriatic Arthritis</strong>. Vital metrics show a low-grade inflammatory state.</p>
            
            <h4>Suggested Next Steps</h4>
            <ul>
                <li><strong>Specialist Referrals:</strong> Consult both a dermatologist for skin management and a rheumatologist to evaluate joint involvement and prevent long-term damage.</li>
                <li><strong>Moisturization:</strong> Apply thick, lipid-rich emollients immediately after bathing to lock in moisture and soften plaques.</li>
                <li><strong>Keratolytic agents:</strong> Salicylic acid preparations can help remove thick scales.</li>
                <li><strong>Joint care:</strong> Gentle stretching, low-impact exercise, and warm compresses can soothe stiff joints.</li>
            </ul>
            
            <h4>Clinical Warning</h4>
            <p>Avoid self-treatment with systemic corticosteroids, as their discontinuation can precipitate life-threatening erythrodermic or pustular psoriasis flares.</p>`
        }
    };

    // --- Tab Switching Logic ---
    const guideButtons = document.querySelectorAll('.guide-tab-btn');
    const guideContents = document.querySelectorAll('.guide-tab-content');

    guideButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;
            
            // Remove active states
            guideButtons.forEach(b => b.classList.remove('active'));
            guideContents.forEach(c => c.classList.remove('active'));
            
            // Add active states
            btn.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        });
    });

    // --- Copy Code Snippet Logic ---
    const copyButtons = document.querySelectorAll('.copy-code-btn');
    copyButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetId = btn.dataset.target;
            const codeEl = document.getElementById(targetId);
            navigator.clipboard.writeText(codeEl.innerText)
                .then(() => {
                    const originalText = btn.innerHTML;
                    btn.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
                    setTimeout(() => {
                        btn.innerHTML = originalText;
                    }, 2000);
                });
        });
    });

    // --- Configuration Modal Management ---
    openConfigBtn.addEventListener('click', () => {
        configModal.classList.add('open');
        // Pre-fill fields from state
        backendModeSelect.value = systemConfig.mode;
        hfEndpointInput.value = systemConfig.hfEndpoint;
        claudeKeyInput.value = systemConfig.claudeKey;
        triggerConfigFieldsVisibility(systemConfig.mode);
    });

    closeConfigBtn.addEventListener('click', () => {
        configModal.classList.remove('open');
    });

    // Close on overlay click
    configModal.addEventListener('click', (e) => {
        if (e.target === configModal) {
            configModal.classList.remove('open');
        }
    });

    backendModeSelect.addEventListener('change', (e) => {
        triggerConfigFieldsVisibility(e.target.value);
    });

    function triggerConfigFieldsVisibility(mode) {
        if (mode === 'live-hf') {
            hfConfigFields.style.display = 'block';
        } else {
            hfConfigFields.style.display = 'none';
        }
    }

    saveConfigBtn.addEventListener('click', () => {
        systemConfig.mode = backendModeSelect.value;
        systemConfig.hfEndpoint = hfEndpointInput.value.trim();
        systemConfig.claudeKey = claudeKeyInput.value.trim();
        
        configModal.classList.remove('open');
        alert(`System configuration updated. Operating in: ${systemConfig.mode === 'live-hf' ? 'Live HF Spaces Mode' : 'Simulation Mode'}`);
    });

    resetConfigBtn.addEventListener('click', () => {
        systemConfig = {
            mode: 'simulated',
            hfEndpoint: '',
            claudeKey: ''
        };
        backendModeSelect.value = 'simulated';
        hfEndpointInput.value = '';
        claudeKeyInput.value = '';
        hfConfigFields.style.display = 'none';
        alert('Configuration reset to Simulated defaults.');
    });

    // --- File Drag & Drop Handlers ---
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = 'var(--color-primary)';
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.style.borderColor = 'rgba(255, 255, 255, 0.15)';
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = 'rgba(255, 255, 255, 0.15)';
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleImageFile(e.dataTransfer.files[0]);
        }
    });

    imageInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleImageFile(e.target.files[0]);
        }
    });

    removeImageBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        clearUploadedImage();
    });

    function handleImageFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid skin lesion or wound image.');
            return;
        }
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            dropzoneDefault.classList.add('hide');
            dropzonePreview.classList.remove('hide');
        };
        reader.readAsDataURL(file);
    }

    function clearUploadedImage() {
        imageInput.value = '';
        imagePreview.src = '#';
        dropzoneDefault.classList.remove('hide');
        dropzonePreview.classList.add('hide');
    }

    // --- Symptom Chips (Toggle Selection) ---
    const symptomChips = document.querySelectorAll('.symptom-chip');
    symptomChips.forEach(chip => {
        chip.addEventListener('click', () => {
            chip.classList.toggle('selected');
        });
    });

    // --- Preset Load Links ---
    const presetButtons = document.querySelectorAll('.preset-btn');
    presetButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const caseKey = btn.dataset.preset;
            loadPresetCase(caseKey);
        });
    });

    function loadPresetCase(key) {
        const caseData = casesPreset[key];
        if (!caseData) return;

        // Vitals
        document.getElementById('patient-age').value = caseData.age;
        document.getElementById('patient-gender').value = caseData.gender;
        document.getElementById('patient-temp').value = caseData.temp;
        document.getElementById('patient-sp02').value = caseData.spo2;
        document.getElementById('patient-hr').value = caseData.hr;
        document.getElementById('patient-bp').value = caseData.bp;

        // Text symptom description
        document.getElementById('symptom-text').value = caseData.text;

        // Symptom chips
        symptomChips.forEach(chip => {
            const val = chip.dataset.val;
            if (caseData.symptoms.includes(val)) {
                chip.classList.add('selected');
            } else {
                chip.classList.remove('selected');
            }
        });

        // Image
        if (caseData.hasImage && caseData.imageMockUrl) {
            imagePreview.src = caseData.imageMockUrl;
            dropzoneDefault.classList.add('hide');
            dropzonePreview.classList.remove('hide');
        } else {
            clearUploadedImage();
        }
    }

    // --- Form Submission / Pipeline Execution ---
    triageForm.addEventListener('submit', (e) => {
        e.preventDefault();
        
        // Show scanning screen
        scanningOverlay.classList.remove('hide');
        resultPlaceholder.classList.add('hide');
        resultContent.classList.add('hide');
        
        let progress = 0;
        const statusSteps = [
            'Parsing natural language features...',
            'Evaluating DistilBERT text embedding weights...',
            'Processing visual patch via EfficientNetB0...',
            'Computing XGBoost patient risk matrix...',
            'Generating SHAP explainability trees...',
            'Orchestrating Claude Agentic advice...'
        ];
        
        const scanTimer = setInterval(() => {
            if (progress < statusSteps.length) {
                scanStatusText.innerText = statusSteps[progress];
                progress++;
            } else {
                clearInterval(scanTimer);
                // Finish scan
                scanningOverlay.classList.add('hide');
                
                if (systemConfig.mode === 'live-hf' && systemConfig.hfEndpoint) {
                    executeLivePipeline();
                } else {
                    executeSimulatedPipeline();
                }
            }
        }, 600);
    });

    // --- Simulated Pipeline Run ---
    function executeSimulatedPipeline() {
        // Collect form data to see if we match any preset or need to generate dynamically
        const age = parseInt(document.getElementById('patient-age').value);
        const spo2 = parseInt(document.getElementById('patient-sp02').value);
        const temp = parseFloat(document.getElementById('patient-temp').value);
        const hr = parseInt(document.getElementById('patient-hr').value);
        const textDesc = document.getElementById('symptom-text').value.toLowerCase();
        const hasImage = !dropzonePreview.classList.contains('hide');

        let selectedCase = null;

        // Pattern matching on description to select closest preset case
        if (textDesc.includes('anaphyl') || textDesc.includes('breathing') || spo2 < 90) {
            selectedCase = casesPreset.emergency;
        } else if (textDesc.includes('psorias') || textDesc.includes('elbow') || textDesc.includes('joint')) {
            selectedCase = casesPreset.psoriasis;
        } else {
            // Default to dermatitis or construct customized metrics
            selectedCase = casesPreset.rash;
        }

        // Render Severity banner
        severityBanner.className = `glass-panel result-card severity-card ${selectedCase.severity}`;
        severityStatusText.innerText = `${selectedCase.severity.toUpperCase()} RISK SEVERITY`;
        xgboostRiskScore.innerText = `${selectedCase.risk}%`;
        clinicalPathway.innerText = selectedCase.pathway;
        urgencyGate.innerText = selectedCase.gate;
        
        if (selectedCase.severity === 'severe') {
            severityBanner.querySelector('.severity-badge-icon').innerHTML = '<i class="fa-solid fa-circle-exclamation"></i>';
        } else if (selectedCase.severity === 'moderate') {
            severityBanner.querySelector('.severity-badge-icon').innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
        } else {
            severityBanner.querySelector('.severity-badge-icon').innerHTML = '<i class="fa-solid fa-shield-halved"></i>';
        }

        // Render Probabilities
        renderProbabilities(selectedCase.textProbs, textPrimaryDisease, textProbBar, textPredictionsList);
        
        if (hasImage && selectedCase.imageProbs) {
            imageClassifierGroup.classList.remove('hide');
            renderProbabilities(selectedCase.imageProbs, imagePrimaryDisease, imageProbBar, imagePredictionsList);
        } else {
            imageClassifierGroup.classList.add('hide');
        }

        // Render SHAP features
        renderSHAP(selectedCase.shap);

        // Render Claude Output (typing effect)
        claudeCacheIndicator.classList.remove('hide');
        typeMarkdownText(selectedCase.claudeAdvice, claudeTextResponse);

        resultContent.classList.remove('hide');
    }

    // --- Live HF Space Pipeline Integration ---
    function executeLivePipeline() {
        const age = parseInt(document.getElementById('patient-age').value);
        const gender = document.getElementById('patient-gender').value;
        const temp = parseFloat(document.getElementById('patient-temp').value);
        const spo2 = parseInt(document.getElementById('patient-sp02').value);
        const hr = parseInt(document.getElementById('patient-hr').value);
        const bp = parseInt(document.getElementById('patient-bp').value);
        const text = document.getElementById('symptom-text').value;
        
        // Extract selected chips
        const activeChips = [];
        document.querySelectorAll('.symptom-chip.selected').forEach(c => {
            activeChips.push(c.dataset.val);
        });

        // Convert uploaded image to base64 if available
        let base64Image = null;
        if (!dropzonePreview.classList.contains('hide')) {
            base64Image = imagePreview.src;
        }

        const requestBody = {
            data: [
                text,
                base64Image,
                age,
                gender,
                temp,
                spo2,
                hr,
                bp,
                activeChips.join(',')
            ]
        };

        fetch(systemConfig.hfEndpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        })
        .then(response => {
            if (!response.ok) throw new Error('API request failed');
            return response.json();
        })
        .then(data => {
            // Gradio standard output format
            const output = data.data; 
            const severity = output[0]; // 'mild', 'moderate', 'severe'
            const riskScore = output[1]; // e.g. 45
            const pathway = output[2]; // 'Standard Triage' or 'Emergency'
            const gate = output[3]; // 'Green', 'Yellow', 'Red'
            const textProbs = output[4]; // Array of {name, pct}
            const imageProbs = output[5]; // Array of {name, pct} or null
            const shapValues = output[6]; // Array of {feature, val, dir}
            const claudeAdvice = output[7]; // markdown string

            // Render Severity banner
            severityBanner.className = `glass-panel result-card severity-card ${severity}`;
            severityStatusText.innerText = `${severity.toUpperCase()} RISK SEVERITY`;
            xgboostRiskScore.innerText = `${riskScore}%`;
            clinicalPathway.innerText = pathway;
            urgencyGate.innerText = gate;

            // Render Probabilities
            renderProbabilities(textProbs, textPrimaryDisease, textProbBar, textPredictionsList);
            if (imageProbs && imageProbs.length > 0) {
                imageClassifierGroup.classList.remove('hide');
                renderProbabilities(imageProbs, imagePrimaryDisease, imageProbBar, imagePredictionsList);
            } else {
                imageClassifierGroup.classList.add('hide');
            }

            // Render SHAP
            renderSHAP(shapValues);

            // Render Claude Output
            claudeCacheIndicator.classList.add('hide');
            typeMarkdownText(claudeAdvice, claudeTextResponse);

            resultContent.classList.remove('hide');
        })
        .catch(err => {
            console.error(err);
            alert('Failed to connect to the Hugging Face Space API endpoint. Reverting to Simulation Mode.');
            executeSimulatedPipeline();
        });
    }

    // Helpers
    function renderProbabilities(probs, primaryEl, barEl, listEl) {
        if (!probs || probs.length === 0) return;
        
        primaryEl.innerText = `${probs[0].name} (${probs[0].pct}%)`;
        barEl.style.width = `${probs[0].pct}%`;
        
        listEl.innerHTML = '';
        probs.forEach(item => {
            const row = document.createElement('div');
            row.className = 'breakdown-row';
            row.innerHTML = `
                <span class="breakdown-label">
                    <span class="status-indicator" style="background-color: var(--color-primary)"></span>
                    ${item.name}
                </span>
                <span class="breakdown-pct">${item.pct}%</span>
            `;
            listEl.appendChild(row);
        });
    }

    function renderSHAP(shapArr) {
        shapChartContainer.innerHTML = '';
        if (!shapArr) return;

        shapArr.forEach(item => {
            const row = document.createElement('div');
            row.className = 'shap-bar-row';
            
            const normalizedWidth = Math.min(Math.abs(item.val) * 8, 40); // Scaling logic
            const widthPct = `${normalizedWidth}%`;
            
            row.innerHTML = `
                <div class="shap-feature-label">${item.feature}</div>
                <div class="shap-bar-track">
                    <div class="shap-bar-midline"></div>
                    <div class="shap-bar-fill ${item.dir}" style="width: ${widthPct}; --width: ${widthPct}"></div>
                    <div class="shap-value-text">${item.val > 0 ? '+' : ''}${item.val.toFixed(1)}</div>
                </div>
            `;
            shapChartContainer.appendChild(row);
        });
    }

    function typeMarkdownText(text, targetEl) {
        targetEl.innerHTML = '';
        
        // Simple element container to append chunks
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = text;
        const childNodes = Array.from(tempDiv.childNodes);
        
        let childIdx = 0;
        function appendNextNode() {
            if (childIdx < childNodes.length) {
                targetEl.appendChild(childNodes[childIdx].cloneNode(true));
                childIdx++;
                setTimeout(appendNextNode, 100);
            }
        }
        appendNextNode();
    }

    // --- Canvas-Based 3D DNA Helix Animation ---
    function initDNACanvas() {
        const canvas = document.getElementById('dna-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const W = canvas.width, H = canvas.height;
        const cx = W / 2;
        const numRungs = 22;
        const helixRadius = 90;
        const rungSpacing = H / (numRungs + 1);
        let angle = 0;

        // Color stops for the helix strands
        const colorA = (t) => {
            // Purple to violet
            const r = Math.round(124 + t * 60);
            const g = Math.round(58 + t * 10);
            const b = Math.round(237 - t * 30);
            return `rgb(${r},${g},${b})`;
        };
        const colorB = (t) => {
            // Pink to rose
            const r = Math.round(192 + t * 30);
            const g = Math.round(132 - t * 20);
            const b = Math.round(252 - t * 60);
            return `rgb(${r},${g},${b})`;
        };

        function drawFrame() {
            ctx.clearRect(0, 0, W, H);

            const rungs = [];
            for (let i = 0; i < numRungs; i++) {
                const t = i / (numRungs - 1);
                const theta = angle + t * Math.PI * 4; // two full twists
                const y = rungSpacing * (i + 1);
                const xA = cx + Math.sin(theta) * helixRadius;
                const xB = cx + Math.sin(theta + Math.PI) * helixRadius;
                const zA = Math.cos(theta);
                const zB = Math.cos(theta + Math.PI);
                rungs.push({ y, xA, xB, zA, zB, t });
            }

            // Draw back-layer rung connectors first (zA < 0)
            rungs.forEach(({ y, xA, xB, zA, zB, t }) => {
                if (zA < 0 && zB < 0) {
                    ctx.beginPath();
                    ctx.moveTo(xA, y);
                    ctx.lineTo(xB, y);
                    ctx.strokeStyle = 'rgba(167,139,250,0.18)';
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                }
            });

            // Draw strand A (behind when zA < 0)
            ctx.beginPath();
            rungs.forEach(({ y, xA, zA }, i) => {
                if (zA < 0) i === 0 ? ctx.moveTo(xA, y) : ctx.lineTo(xA, y);
            });
            ctx.strokeStyle = 'rgba(124,58,237,0.25)';
            ctx.lineWidth = 3;
            ctx.stroke();

            // Draw strand B (behind when zB < 0)
            ctx.beginPath();
            rungs.forEach(({ y, xB, zB }, i) => {
                if (zB < 0) i === 0 ? ctx.moveTo(xB, y) : ctx.lineTo(xB, y);
            });
            ctx.strokeStyle = 'rgba(192,132,252,0.25)';
            ctx.lineWidth = 3;
            ctx.stroke();

            // Draw front-layer rung connectors
            rungs.forEach(({ y, xA, xB, zA, zB, t }) => {
                if (zA >= 0 || zB >= 0) {
                    ctx.beginPath();
                    ctx.moveTo(xA, y);
                    ctx.lineTo(xB, y);
                    ctx.strokeStyle = `rgba(167,139,250,${0.3 + t * 0.3})`;
                    ctx.lineWidth = 1.8;
                    ctx.stroke();
                }
            });

            // Draw strand A (front)
            ctx.beginPath();
            let moved = false;
            rungs.forEach(({ y, xA, zA }) => {
                if (zA >= 0) {
                    if (!moved) { ctx.moveTo(xA, y); moved = true; }
                    else ctx.lineTo(xA, y);
                } else { moved = false; }
            });
            ctx.strokeStyle = '#7c3aed';
            ctx.lineWidth = 4;
            ctx.lineJoin = 'round';
            ctx.stroke();

            // Draw strand B (front)
            ctx.beginPath();
            moved = false;
            rungs.forEach(({ y, xB, zB }) => {
                if (zB >= 0) {
                    if (!moved) { ctx.moveTo(xB, y); moved = true; }
                    else ctx.lineTo(xB, y);
                } else { moved = false; }
            });
            ctx.strokeStyle = '#c084fc';
            ctx.lineWidth = 4;
            ctx.stroke();

            // Draw nodes on both strands
            rungs.forEach(({ y, xA, xB, zA, zB, t }) => {
                const sA = Math.max(0.4, zA * 0.5 + 0.7);
                const rA = 5 + zA * 3;
                if (rA > 0) {
                    ctx.beginPath();
                    ctx.arc(xA, y, rA, 0, Math.PI * 2);
                    ctx.fillStyle = colorA(t);
                    ctx.globalAlpha = sA;
                    ctx.fill();

                    // Glow
                    const glowA = ctx.createRadialGradient(xA, y, 0, xA, y, rA * 2.5);
                    glowA.addColorStop(0, 'rgba(124,58,237,0.4)');
                    glowA.addColorStop(1, 'rgba(124,58,237,0)');
                    ctx.beginPath();
                    ctx.arc(xA, y, rA * 2.5, 0, Math.PI * 2);
                    ctx.fillStyle = glowA;
                    ctx.globalAlpha = sA * 0.6;
                    ctx.fill();
                }
                ctx.globalAlpha = 1;

                const sB = Math.max(0.4, zB * 0.5 + 0.7);
                const rB = 5 + zB * 3;
                if (rB > 0) {
                    ctx.beginPath();
                    ctx.arc(xB, y, rB, 0, Math.PI * 2);
                    ctx.fillStyle = colorB(t);
                    ctx.globalAlpha = sB;
                    ctx.fill();

                    const glowB = ctx.createRadialGradient(xB, y, 0, xB, y, rB * 2.5);
                    glowB.addColorStop(0, 'rgba(192,132,252,0.4)');
                    glowB.addColorStop(1, 'rgba(192,132,252,0)');
                    ctx.beginPath();
                    ctx.arc(xB, y, rB * 2.5, 0, Math.PI * 2);
                    ctx.fillStyle = glowB;
                    ctx.globalAlpha = sB * 0.6;
                    ctx.fill();
                }
                ctx.globalAlpha = 1;
            });

            angle += 0.012;
            requestAnimationFrame(drawFrame);
        }
        drawFrame();
    }
    initDNACanvas();
});
