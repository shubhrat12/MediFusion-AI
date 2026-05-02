# Advanced Medical AI Analysis & Prescription Platform
## Project Overview
A cutting-edge medical diagnostic system leveraging state-of-the-art deep learning models and Large Language Models (LLM) for comprehensive patient analysis. This revolutionary platform combines medical imaging, natural language processing, and structured data analysis to deliver comprehensive medical assessments and tailored prescription recommendations.

## Core Components

### 1. Medical Image Analysis Engine (CheXpert Transfer Learning)
- **Base Model**: DenseNet-121 pretrained on ImageNet
- **Transfer Learning Process**:
  1. Initial Training on Stanford's CheXpert dataset (224,316 chest radiographs)
  2. Domain adaptation using custom loss functions
  3. [Final transfer learning on the target CheXpert small X-ray dataset](https://www.kaggle.com/datasets/ashery/chexpert)
- **Model Evaluation**:
  - **After Transfer Learning**:
    - Accuracy: 89.5%
    - F1-Score: 0.87
    - AUC-ROC: 0.92
    - Precision: 0.88
    - Recall: 0.86


### 2. Advanced NLP System (Clinical BERT Transfer Learning)
- **Base Model**: BioBERT pretrained on PubMed abstracts
- **Transfer Learning Pipeline**:
  1. Initial pretraining on 2M medical documents
  2. Domain adaptation on 100K radiology reports
  3. [Final transfer learning on the custom dataset](https://www.mediafire.com/file/3gornmrb27ftbzv/Dataset.zip/file)

- **Architecture**: 
  - 12 transformer layers
  - 768 hidden dimensions
  - 12 attention heads
  - Medical vocabulary expansion: +18,000 tokens
- **Performance Evaluation**:
  - **After Transfer Learning**:
    - Accuracy: 67.1%
    - F1-Score: 0.87
    - Precision: 0.84
    - Recall: 0.91


### 3. Multi-Disease Classification System (Trained From Scratch)
Ensemble of specialized models for critical disease detection, each trained on carefully curated datasets:

#### a. Diabetes Prediction Model
- **Architecture**: Custom Gradient Boosting Classifier
- **Dataset**: [BRFSS2015 Diabetes Dataset (253,680 samples)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

- **Feature Engineering**:
  - 16 vital health markers optimization
  - Advanced correlation analysis
  - Custom feature scaling pipeline
- **Model Performance**:
  - Accuracy: 75.2%
  - Precision: 0.73
  - Recall: 0.79
  - F1-Score: 0.76
  - AUC-ROC: 0.83

#### b. Kidney Disease Analytics
- **Architecture**: Enhanced Random Forest with Custom Preprocessing
- **Dataset**: [Chronic Kidney Disease Dataset (400 samples)](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)
- **Feature Processing**:
  - 24 biological markers
  - Missing value imputation
  - Advanced feature selection
- **Model Performance**:
  - Accuracy: 97.5%
  - Precision: 1.0
  - Recall: 0.93
  - F1-Score: 0.96
  - AUC-ROC: 1.0

#### c. Heart Disease Detection
- **Architecture**: Deep Neural Network
  - Input Layer: 13 nodes
  - Hidden Layers: [256, 128, 64] nodes
  - Activation: ReLU + Batch Normalization
  - Dropout: 0.3
- **Dataset**: [Cleveland Heart Disease Dataset (303 samples)](https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland)

- **Model Performance**:
  - Accuracy: 88.5%
  - Precision: 0.83
  - Recall: 0.92
  - F1-Score: 0.88
  - AUC-ROC: 0.97
    
### 4. LLM-Powered Diagnostic Fusion & Prescription System
- **Model**: Locally-hosted Llama2 LLM
- **Integration**: Custom prompt engineering for medical context
- **Capabilities**:
  - Multi-modal result fusion
  - Contextual prescription generation
  - Drug interaction analysis
  - Patient-specific medication adjustments
- **Features**:
  - Considers comorbidities in prescriptions
  - Real-time medication compatibility checks
  - Dosage optimization based on patient conditions

### 5. Interactive GUI System
- Modern PyQt5-based interface
- Drag-and-drop functionality for X-rays
- Real-time analysis feedback
- Integrated report generation
- Export capabilities for medical records
## Project Structure
```
├── Image/          # Image analysis models and scripts
├── Input/          # Sample input data
├── Scripts/        # Core Python processing scripts  
├── Table/          # Tabular data models
│   ├── Diabetes/
│   ├── Heart_Disease/
│   └── Kidney/
└── Text/           # Text analysis models
```

## Setup & Installation
1. Create a Python virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

3. Run the application:
```powershell
python gui.py
```
**Note**: Your system is expected to have Llama 3.2 installed locally.  
If not, please [download it here](https://www.llama.com/llama-downloads/).



## Development Guidelines
1. Security:
   - Never commit credentials or tokens
   - Use only relative paths
   - Validate all user inputs
   - Properly handle errors
   - Follow least privilege principle

2. Data Protection:
   - Use anonymized test data only
   - Keep models and checkpoints local
   - Clean output files of metadata
   
3. Code Quality:
   - Follow PEP 8 style guide
   - Add proper documentation
   - Include error handling
   - Write unit tests


## Technical Architecture

### Deep Learning Infrastructure
- **Framework**: PyTorch with CUDA acceleration
- **Computing**: GPU-optimized for real-time inference
- **Memory Management**: Efficient batch processing
- **Parallelization**: Multi-threaded data processing

### 1. Core Pipeline Architecture

The primary logic is implemented in `run_all_models.py` via the `MedicalAIPipeline` class. It integrates multiple AI models to perform comprehensive patient analysis and treatment recommendations.

#### A. Multi-Modal Input Processing

##### Image Analysis
- Inputs chest X-ray images
- Uses a DenseNet121 model with attention mechanisms
- Performs multi-label classification for 13 lung diseases
- Implementation: `train_chexagent_model.py` and related files

##### Text Analysis
- Inputs radiology reports
- Uses a BERT-based model fine-tuned on clinical text
- Performs multi-label classification matching image model's outputs
- Includes a fallback keyword-matching system

##### Tabular Data Analysis
- **Diabetes**: XGBoost model on 21 features
- **Kidney Disease**: Random Forest on 24 features
- **Heart Disease**: MLP classifier using clinical inputs
- All provide binary classification (Yes/No) with probability outputs

#### B. Model Integration

##### Disease Detection Fusion
- Merges predictions from image and text models
- Resolves conflicts using confidence scores

##### Risk Assessment
- Combines heart, kidney, and diabetes results
- Builds a complete comorbidity risk profile

#### C. LLM Integration

- Takes merged predictions from all models
- Generates structured medical prompts
- Uses a local LLaMA model for recommendations
- Considers comorbidities and contraindications
- Implements fallback systems (Ollama, GPT4All, LM Studio)

### 2. LLM Fusion Strategy

The LLM layer acts as a high-level decision-making system that:

#### Aggregates Outputs
- Integrates predictions from image, text, and tabular models
- Utilizes confidence scores to balance conflicting results

#### Contextual Reasoning
- Analyzes relations between multiple diseases
- Evaluates risks posed by comorbidities like diabetes, heart, and kidney disease

#### Medical Knowledge Application
- Applies clinical logic to suggest compatible treatments
- Highlights drug contraindications based on comorbidity status
- Adjusts or recommends alternative medications accordingly

#### Final Decision Support
- Prioritizes treatment based on severity
- Balances multiple therapies to avoid medical conflict
- Produces a holistic treatment recommendation plan

## Diagramatic Representation
<img width="1536" height="1024" alt="447408689-9d06f2c2-81bd-4a44-823b-5d6763790725" src="https://github.com/user-attachments/assets/c42025f0-2e9d-49e0-9045-2960d319631d" />



## Usage Example


https://github.com/user-attachments/assets/e728610c-a14f-453c-ae37-21fc6844bb09







## Implementation Highlights

### Advanced Model Training
- **Transfer Learning Pipeline**:
  - Pre-trained on large medical datasets
  - Fine-tuned on specialized conditions
  - Custom loss functions for medical context

### Innovative LLM Integration
- **Custom Medical Knowledge Base**:
  - Drug interaction database
  - Disease comorbidity patterns
  - Treatment protocols
- **Context-Aware Processing**:
  - Patient history consideration
  - Medication contradiction prevention
  - Dynamic dosage adjustment

### Scalable Architecture
- Modular design for easy updates
- Parallel processing capabilities
- CPU/GPU flexible deployment
- Containerization support

## Future Enhancements
1. Integration with Electronic Health Records
2. Mobile application development
3. Cloud deployment options
4. Additional disease modules
5. Enhanced prescription automation
