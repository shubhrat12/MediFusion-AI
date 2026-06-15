# AI-Driven Medical Diagnosis & Treatment Recommendation System
## Overview
A next-generation clinical decision-support platform that harnesses deep learning and Large Language Models (LLMs) to conduct thorough patient evaluations. This groundbreaking system unifies medical imaging interpretation, natural language understanding, and structured health data processing to generate complete diagnostic assessments alongside personalized treatment and prescription plans.

---

## System Components

### 1. Chest X-Ray Interpretation Engine (CheXpert Transfer Learning)
- **Backbone Architecture**: DenseNet-121 initialized with ImageNet weights
- **Transfer Learning Stages**:
  1. Primary training on Stanford's CheXpert dataset (224,316 chest radiographs)
  2. Domain-specific adaptation using custom-designed loss functions
  3. [Final fine-tuning on the CheXpert small X-ray dataset](https://www.kaggle.com/datasets/ashery/chexpert)
- **Post-Transfer-Learning Performance**:
  - Accuracy: 89.5%
  - F1-Score: 0.87
  - AUC-ROC: 0.92
  - Precision: 0.88
  - Recall: 0.86

---

### 2. Clinical NLP Engine (Clinical BERT Transfer Learning)
- **Foundation Model**: BioBERT pre-trained on PubMed abstracts
- **Transfer Learning Stages**:
  1. Continued pre-training on 2 million medical documents
  2. Adaptation on 100,000 radiology reports
  3. [Final fine-tuning on the custom dataset](https://www.mediafire.com/file/3gornmrb27ftbzv/Dataset.zip/file)
- **Model Architecture**:
  - 12 transformer encoder layers
  - 768-dimensional hidden states
  - 12 self-attention heads
  - Medical vocabulary expanded by +18,000 tokens
- **Post-Transfer-Learning Performance**:
  - Accuracy: 67.1%
  - F1-Score: 0.87
  - Precision: 0.84
  - Recall: 0.91

---

### 3. Specialized Disease Classification Suite (Trained From Scratch)
A set of purpose-built models targeting high-priority conditions, each trained on carefully assembled datasets:

#### a. Diabetes Risk Predictor
- **Algorithm**: Custom Gradient Boosting Classifier
- **Training Data**: [BRFSS2015 Diabetes Health Indicators Dataset (253,680 records)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- **Feature Engineering**:
  - Optimization across 16 vital health markers
  - Deep correlation analysis between variables
  - Custom feature normalization pipeline
- **Performance Metrics**:
  - Accuracy: 75.2%
  - Precision: 0.73
  - Recall: 0.79
  - F1-Score: 0.76
  - AUC-ROC: 0.83

#### b. Chronic Kidney Disease Analyzer
- **Algorithm**: Enhanced Random Forest with Tailored Preprocessing
- **Training Data**: [Chronic Kidney Disease Dataset (400 samples)](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)
- **Feature Handling**:
  - 24 biological and lab markers
  - Automated missing value imputation
  - Rigorous feature selection methodology
- **Performance Metrics**:
  - Accuracy: 97.5%
  - Precision: 1.0
  - Recall: 0.93
  - F1-Score: 0.96
  - AUC-ROC: 1.0

#### c. Cardiac Disease Detector
- **Algorithm**: Fully Connected Deep Neural Network
  - Input Layer: 13 nodes
  - Hidden Layers: [256, 128, 64] neurons
  - Activation Function: ReLU + Batch Normalization
  - Regularization: Dropout at 0.3
- **Training Data**: [Cleveland Heart Disease Dataset (303 samples)](https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland)
- **Performance Metrics**:
  - Accuracy: 88.5%
  - Precision: 0.83
  - Recall: 0.92
  - F1-Score: 0.88
  - AUC-ROC: 0.97

---

### 4. LLM-Powered Diagnostic Fusion & Prescription Engine
- **Model**: Locally-deployed Llama2 LLM
- **Integration Method**: Specialized medical prompt engineering
- **Core Capabilities**:
  - Cross-modal result synthesis and fusion
  - Context-sensitive prescription generation
  - Drug interaction screening and analysis
  - Patient-condition-aware medication adjustments
- **Key Features**:
  - Full comorbidity awareness during prescription
  - Live medication compatibility verification
  - Condition-specific dosage optimization

---

### 5. Graphical User Interface
- Developed with PyQt5 for a modern look and feel
- Drag-and-drop support for loading X-ray images
- Live feedback during analysis and inference
- Integrated medical report generation module
- Export-ready output for electronic health records

---

## Repository Layout
```
├── Image/          # Imaging models and associated processing scripts
├── Input/          # Example/sample input data
├── Scripts/        # Core Python processing and pipeline scripts
├── Table/          # Tabular machine learning models
│   ├── Diabetes/
│   ├── Heart_Disease/
│   └── Kidney/
└── Text/           # Text classification models
```

---

## Installation & Setup

1. Create and activate a Python virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install all required packages:
```powershell
pip install -r requirements.txt
```

3. Launch the application:
```powershell
python gui.py
```
**Important**: Llama 3.2 must be installed locally on your machine.  
If it isn't already, [download it here](https://www.llama.com/llama-downloads/).

---

## Development Standards

### Security
- Never store credentials or tokens in version control
- Use only relative file paths throughout the codebase
- Sanitize and validate all user inputs
- Implement thorough error handling at every layer
- Follow the principle of least privilege at all times

### Data Protection
- Use anonymized or synthetic data exclusively for testing
- Keep model weights and checkpoints stored locally
- Strip all metadata from exported output files

### Code Quality
- Adhere to the PEP 8 Python style guide
- Provide clear inline documentation and docstrings
- Include meaningful error handling throughout
- Write and maintain unit tests for all core functionality

---

## Technical Architecture

### Deep Learning Infrastructure
- **Framework**: PyTorch with CUDA acceleration
- **Compute**: GPU-optimized for real-time model inference
- **Memory**: Efficient batch processing to minimize overhead
- **Concurrency**: Multi-threaded data loading and preprocessing

---

### Core Pipeline Architecture

The central orchestration logic lives in `run_all_models.py`, implemented as the `MedicalAIPipeline` class. It coordinates all AI subsystems to deliver a unified patient analysis and treatment recommendation.

#### A. Multi-Modal Input Processing

##### Imaging
- Accepts chest X-ray files as input
- Processed by a DenseNet121 model with attention mechanisms
- Outputs multi-label classification across 13 lung pathologies
- Implemented in `train_chexagent_model.py` and companion scripts

##### Clinical Text
- Accepts free-text radiology reports as input
- Processed by a fine-tuned BERT-based clinical NLP model
- Outputs multi-label predictions aligned with imaging model outputs
- Falls back to keyword-matching when model confidence is low

##### Structured Tabular Data
- **Diabetes**: XGBoost model operating on 21 engineered features
- **Kidney Disease**: Random Forest trained on 24 clinical features
- **Heart Disease**: MLP classifier using 13 standard clinical inputs
- All three output binary classification (Positive/Negative) with probability scores

#### B. Model Integration & Fusion

##### Disease Detection Fusion
- Combines predictions from both imaging and text models
- Resolves disagreements between modalities by weighting confidence scores

##### Comorbidity Risk Profiling
- Aggregates outputs from heart, kidney, and diabetes classifiers
- Assembles a full comorbidity risk map for each patient

#### C. LLM Integration Layer

- Receives fused outputs from all upstream models
- Constructs structured medical prompts for the language model
- Invokes a locally running LLaMA instance for recommendation generation
- Accounts for comorbidities and known contraindications
- Supports fallback LLM backends: Ollama, GPT4All, and LM Studio

---

### LLM Fusion Strategy

The LLM operates as a high-level reasoning layer that synthesizes all model outputs into actionable clinical guidance:

#### Output Aggregation
- Integrates predictions from imaging, text, and tabular pipelines
- Applies confidence-based weighting to reconcile conflicting signals

#### Contextual Medical Reasoning
- Examines disease interrelationships across detected conditions
- Assesses how comorbidities such as diabetes, heart disease, and chronic kidney disease affect one another

#### Clinical Knowledge Application
- Applies domain logic to propose compatible treatment strategies
- Identifies drug contraindications arising from specific comorbidity combinations
- Suggests alternative medications when standard therapies are contraindicated

#### Treatment Prioritization & Synthesis
- Ranks treatment urgency based on severity of detected conditions
- Balances multiple concurrent therapies to avoid harmful interactions
- Produces a unified, patient-centered treatment recommendation

---

## System Architecture Diagram

<img width="1536" height="1024" alt="447408689-9d06f2c2-81bd-4a44-823b-5d6763790725" src="https://github.com/user-attachments/assets/c42025f0-2e9d-49e0-9045-2960d319631d" />

---

## Demo

https://github.com/user-attachments/assets/e728610c-a14f-453c-ae37-21fc6844bb09

---

## Implementation Highlights

### Robust Model Training Pipeline
- **Transfer Learning Approach**:
  - Pre-trained on large, publicly available medical datasets
  - Fine-tuned on condition-specific, curated subsets
  - Custom loss functions designed for the clinical prediction context

### Sophisticated LLM Integration
- **Medical Knowledge Base**:
  - Comprehensive drug interaction reference database
  - Established disease comorbidity co-occurrence patterns
  - Evidence-based clinical treatment protocols
- **Context-Sensitive Reasoning**:
  - Full consideration of patient history during inference
  - Proactive medication contradiction prevention
  - Dynamic dosage recommendation based on patient state

### Modular & Scalable Design
- Component-based architecture enabling independent module updates
- Support for parallel processing across pipeline stages
- Flexible deployment on either CPU-only or GPU-accelerated hardware
- Containerization-ready for cloud or on-premise deployment

---

## Planned Future Enhancements
1. Electronic Health Record (EHR) system integration
2. Native mobile application development
3. Cloud-hosted deployment options
4. Additional disease detection modules
5. Further automation of the prescription generation workflow
