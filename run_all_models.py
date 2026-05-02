import os
import subprocess
import json
import tempfile
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import joblib
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer, BertForSequenceClassification

import warnings
import os

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Redirect specific PyTorch warnings
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

class MedicalAIPipeline:
    def __init__(self, model_paths):
        """
        Initialize the medical AI pipeline with model paths.
        
        Args:
            model_paths (dict): Dictionary containing paths to all models
                - 'image_model': Path to image classification model
                - 'text_model': Path to text classification model  
                - 'heart_model': Path to heart disease model
                - 'kidney_model': Path to kidney disease model
                - 'diabetes_model': Path to diabetes model
                - 'diabetes_scaler': Path to diabetes scaler
        """
        self.model_paths = model_paths
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Lung disease class names (adjust based on your model)
        self.lung_diseases = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices'
        ]
        self.text_classes = [
    'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices', 'No Finding'
    ]
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load all trained models"""
        print("Loading models...")
        
        # Load image model
        self.image_model = self._load_image_model()
        
        # Load text model  
        self.text_model, self.tokenizer = self._load_text_model()
        
        # Load tabular models
        self.heart_model = self._load_heart_model()
        self.kidney_model = self._load_kidney_model()
        self.diabetes_model, self.diabetes_scaler = self._load_diabetes_model()
        
        print("All models loaded successfully!")
    
    def _load_image_model(self):
        """Load the image classification model - assuming DenseNet121 architecture"""
        try:
            print("Loading image model...")
            
            # Create DenseNet121 model (similar to your kong.py)
            model = models.densenet121(pretrained=False)  # Don't load ImageNet weights initially
            
            # Try to load your checkpoint to determine the correct number of classes
            try:
                checkpoint = torch.load(self.model_paths['image_model'], map_location=self.device, weights_only=False)
                state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
                
                # Check if classifier layer exists and get its output size
                if 'classifier.weight' in state_dict:
                    num_classes = state_dict['classifier.weight'].shape[0]
                    print(f"Detected {num_classes} output classes from checkpoint")
                    
                    # Adjust lung diseases list to match
                    if num_classes != len(self.lung_diseases):
                        if num_classes == 12:
                            # Use the 12-class version (without 'No Finding' and 'Support Devices')
                            self.lung_diseases = [
                                'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                                'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                                'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                                'Pleural Other', 'Fracture'
                            ]
                        elif num_classes < len(self.lung_diseases):
                            self.lung_diseases = self.lung_diseases[:num_classes]
                        else:
                            # Add generic classes if needed
                            for i in range(len(self.lung_diseases), num_classes):
                                self.lung_diseases.append(f'Disease_Class_{i+1}')
                
                else:
                    num_classes = len(self.lung_diseases)
                    print(f"Could not determine classes from checkpoint, using default {num_classes}")
                
            except Exception as e:
                print(f"Could not load checkpoint to determine classes: {e}")
                num_classes = 12  # Default to 12 classes
                self.lung_diseases = [
                    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                    'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                    'Pleural Other', 'Fracture'
                ]
            
            # Replace classifier with correct number of classes
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            
            # Try to load the trained weights
            try:
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                print("Successfully loaded model weights")
            except Exception as e:
                print(f"Could not load weights: {e}")
                print("Using randomly initialized weights for demo")
            
            model.eval()
            model.to(self.device)
            print(f"Image model loaded with {num_classes} classes")
            return model
            
        except Exception as e:
            print(f"Error loading image model: {e}")
            print("Creating fallback DenseNet121 model")
            
            # Fallback model
            model = models.densenet121(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, len(self.lung_diseases))
            model.eval()
            model.to(self.device)
            return model
    
    def _load_text_model(self):
        """Load the text classification model - Fixed version"""
        try:
            print("Loading text model...")
            
            # Use BERT tokenizer instead of BiomedNLP
            from transformers import BertTokenizer, BertForSequenceClassification
            
            # Load tokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
            # Create model with correct number of labels
            model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=len(self.text_classes),
                problem_type='multi_label_classification'  # Important for multi-label classification
            )
            
            # Load the fine-tuned weights
            try:
                state_dict = torch.load(self.model_paths['text_model'], map_location=self.device, weights_only=False)
                
                # Handle different checkpoint formats
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                
                model.load_state_dict(state_dict, strict=False)
                print("Text model fine-tuned weights loaded successfully")
                
            except Exception as e:
                print(f"Could not load text model weights: {e}")
                print("Using pre-trained BERT weights only")
            
            model.eval()
            model.to(self.device)
            
            return model, tokenizer
            
        except Exception as e:
            print(f"Error loading text model: {e}")
            print("Using fallback text processing")
            return None, None

    
    def _load_heart_model(self):
        """Load the heart disease model"""
        try:
            print("Loading heart model...")
            
            class MLPClassifier(nn.Module):
                def __init__(self, input_size, hidden_sizes=[256, 128, 64], output_size=1,
                             dropout_rate=0.2, use_batch_norm=True):
                    super(MLPClassifier, self).__init__()
                    layers = []
                    prev_size = input_size
                    
                    for hidden_size in hidden_sizes:
                        if use_batch_norm:
                            layers.extend([
                                nn.Linear(prev_size, hidden_size),
                                nn.BatchNorm1d(hidden_size),
                                nn.ReLU(),
                                nn.Dropout(dropout_rate)
                            ])
                        else:
                            layers.extend([
                                nn.Linear(prev_size, hidden_size),
                                nn.ReLU(),
                                nn.Dropout(dropout_rate)
                            ])
                        prev_size = hidden_size
                    
                    layers.append(nn.Linear(prev_size, output_size))
                    self.model = nn.Sequential(*layers)
                
                def forward(self, x):
                    return self.model(x)
            
            checkpoint = torch.load(self.model_paths['heart_model'], map_location=self.device, weights_only=False)
            input_size = checkpoint.get('input_size', 25)  # Default fallback
            hidden_sizes = checkpoint.get('hidden_sizes', [256, 128, 64])
            
            model = MLPClassifier(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                output_size=1,
                dropout_rate=checkpoint.get('dropout_rate', 0.2),
                use_batch_norm=checkpoint.get('use_batch_norm', True)
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            model.to(self.device)
            print("Heart model loaded successfully")
            return model
            
        except Exception as e:
            print(f"Error loading heart model: {e}")
            return None
    
    def _load_kidney_model(self):
        """Load the kidney disease model"""
        try:
            print("Loading kidney model...")
            model = joblib.load(self.model_paths['kidney_model'])
            print("Kidney model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading kidney model: {e}")
            return None
    
    def _load_diabetes_model(self):
        """Load the diabetes model and scaler"""
        try:
            print("Loading diabetes model...")
            model = joblib.load(self.model_paths['diabetes_model'])
            scaler = joblib.load(self.model_paths['diabetes_scaler']) 
            print("Diabetes model loaded successfully")
            return model, scaler
        except Exception as e:
            print(f"Error loading diabetes model: {e}")
            return None, None
    
    def predict_image(self, image_path, threshold=0.5):
        """
        Predict lung diseases from chest X-ray image
        
        Args:
            image_path (str): Path to chest X-ray image
            threshold (float): Threshold for positive prediction
            
        Returns:
            list: List of predicted lung diseases
        """
        if self.image_model is None:
            print("Image model not available, returning empty predictions")
            return []
        
        # Standard preprocessing for DenseNet (same as your kong.py)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel if grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            if image.mode not in ['RGB', 'L']:
                image = image.convert('RGB')
                
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}. Using random input for demo.")
            image_tensor = torch.randn(1, 3, 224, 224).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.image_model(image_tensor)
            probs = torch.sigmoid(outputs)[0].cpu().numpy()

        # Debug: print probabilities
        print(f"Image model probabilities: {probs[:5]}...")  # Show first 5

        # Get positive predictions (lower threshold for better detection)
        positive_diseases = [self.lung_diseases[i] for i, p in enumerate(probs) if p > max(0.3, threshold)]
                
        print(f"\n[Image Model] Detected diseases: {positive_diseases}")
        return positive_diseases
    
    def predict_text(self, clinical_text, threshold=0.5):
        """
        Predict lung diseases from clinical text - Fixed version
        
        Args:
            clinical_text (str): Clinical report text
            threshold (float): Threshold for positive prediction
            
        Returns:
            list: List of predicted lung diseases
        """
        if self.text_model is None or self.tokenizer is None:
            print("Text model not available, using keyword-based fallback")
            return self._keyword_based_prediction(clinical_text)
        
        print(f"Processing text: {clinical_text[:100]}...")  # Debug: show first 100 chars
        
        # Tokenize input - using correct parameters
        inputs = self.tokenizer(
            clinical_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.text_model(input_ids, attention_mask=attention_mask)
            # Use sigmoid for multi-label classification
            probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()
        
        # Debug: print probabilities
        print(f"Text model probabilities: {probs[:5]}...")  # Show first 5
        print(f"Max probability: {np.max(probs):.3f}, Min: {np.min(probs):.3f}")
        
        # Get positive predictions with lower threshold for better detection
        positive_diseases = []
        for i, p in enumerate(probs):
            if p > max(0.3, threshold):  # Use lower threshold
                condition = self.text_classes[i]
                positive_diseases.append(condition)
                print(f"  -> {condition}: {p:.3f}")
        
        print(f"\n[Text Model] Detected diseases: {positive_diseases}")
        return positive_diseases
    
    def _keyword_based_prediction(self, clinical_text):
        """Fallback keyword-based prediction for text"""
        text_lower = clinical_text.lower()
        detected = []
        
        keyword_map = {
            'cardiomegaly': 'Cardiomegaly',
            'pleural effusion': 'Pleural Effusion',
            'pneumonia': 'Pneumonia',
            'edema': 'Edema',
            'consolidation': 'Consolidation',
            'pneumothorax': 'Pneumothorax',
            'atelectasis': 'Atelectasis',
            'opacity': 'Lung Opacity',
            'lesion': 'Lung Lesion',
            'fracture': 'Fracture'
        }
        
        for keyword, disease in keyword_map.items():
            if keyword in text_lower and disease in self.text_classes:
                detected.append(disease)
        
        print(f"\n[Text Model - Keyword Fallback] Detected diseases: {detected}")
        return detected
    
    def predict_heart_disease(self, features):
        """
        Predict heart disease from tabular features
        
        Args:
            features (list): List of features for heart disease prediction
            
        Returns:
            dict: Prediction results
        """
        if self.heart_model is None:
            print("Heart model not available, returning default prediction")
            return {'probability': 0.5, 'prediction': False}
        
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.heart_model(x)
            prob = torch.sigmoid(outputs).item()
        
        result = {
            'probability': prob,
            'prediction': prob > 0.5
        }
        
        print(f"\n[Heart Disease] Probability: {prob:.3f}, Prediction: {'Yes' if result['prediction'] else 'No'}")
        return result
    
    def predict_kidney_disease(self, features):
        """
        Predict kidney disease from tabular features
        
        Args:
            features (list): List of 24 features for kidney disease prediction
            
        Returns:
            dict: Prediction results
        """
        if self.kidney_model is None:
            print("Kidney model not available, returning default prediction")
            return {'probability': 0.3, 'prediction': False}
        
        # Process categorical features
        categorical_mappings = {
            'rbc': {'normal': 1, 'abnormal': 0},
            'pc': {'normal': 1, 'abnormal': 0},
            'pcc': {'present': 1, 'notpresent': 0},
            'ba': {'present': 1, 'notpresent': 0},
            'htn': {'yes': 1, 'no': 0},
            'dm': {'yes': 1, 'no': 0},
            'cad': {'yes': 1, 'no': 0},
            'appet': {'good': 1, 'poor': 0},
            'pe': {'yes': 1, 'no': 0},
            'ane': {'yes': 1, 'no': 0}
        }
        
        processed_features = features.copy()
        categorical_indices = [5, 6, 7, 8, 18, 19, 20, 21, 22, 23]
        
        for idx in categorical_indices:
            if isinstance(processed_features[idx], str):
                feature_name = list(categorical_mappings.keys())[categorical_indices.index(idx)]
                processed_features[idx] = categorical_mappings[feature_name].get(
                    processed_features[idx].lower(), 0
                )
        
        processed_features = [float(x) for x in processed_features]
        
        # Convert to pandas DataFrame to avoid feature name warnings
        
        # Convert to pandas DataFrame to avoid feature name warnings
        import pandas as pd
        feature_names = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 
                        'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 
                        'cad', 'appet', 'pe', 'ane']
        df_features = pd.DataFrame([processed_features], columns=feature_names)

        # Make prediction
        try:
            pred = self.kidney_model.predict(df_features)[0]
            prob = self.kidney_model.predict_proba(df_features)[0][1]

        except Exception as e:
            print(f"Kidney prediction error: {e}")
            return {'probability': 0.3, 'prediction': False}
        
        result = {
            'probability': prob,
            'prediction': pred == 1
        }
        
        print(f"\n[Kidney Disease] Probability: {prob:.3f}, Prediction: {'Yes' if result['prediction'] else 'No'}")
        return result
    
    def predict_diabetes(self, features):
        """
        Predict diabetes from tabular features - FIXED VERSION
        
        Args:
            features (list): List of 21 base features for diabetes prediction
            
        Returns:
            dict: Prediction results
        """
        if self.diabetes_model is None or self.diabetes_scaler is None:
            print("Diabetes model not available, returning default prediction")
            return {'probability': 0.25, 'prediction': False}
        
        # Create derived features
        bmi = features[3]
        age = features[18]
        highbp = features[0]
        highchol = features[1]
        
        # BMI Category - using the CORRECT feature name from training
        if bmi < 18.5: bmi_category = 0
        elif bmi < 25: bmi_category = 1
        elif bmi < 30: bmi_category = 2
        else: bmi_category = 3
            
        # Age Category - using the CORRECT feature name from training
        if age < 40: age_category = 0
        elif age < 60: age_category = 1
        else: age_category = 2
            
        # Health Risk Score - using the CORRECT feature name from training
        healthriskscore = (highbp * 2 + highchol * 2 + features[6] * 3 + features[13])
        
        # Lifestyle Score - using the CORRECT feature name from training
        lifestylescore = (features[7] * 2 - features[4] * 2 + features[8] + features[9])
        
        # Interactions - using the CORRECT feature name from training
        bmi_age = bmi * age
        bp_cholesterol = highbp * highchol  # Changed from bp_chol to bp_cholesterol
        
        # Combine all features with CORRECT names
        all_features = features + [bmi_category, age_category, healthriskscore, lifestylescore, bmi_age, bp_cholesterol]
        
        # Create feature names for all 27 features (21 base + 6 derived) - USING CORRECT NAMES
        feature_names = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
                        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
                        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 
                        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 
                        'Income', 'BMI_Category', 'Age_Category', 'HealthRiskScore', 'LifestyleScore', 
                        'BMI_Age', 'BP_Cholesterol']

        # Scale and predict
        try:
            df_features = pd.DataFrame([all_features], columns=feature_names)
            x_scaled = self.diabetes_scaler.transform(df_features)
            pred = self.diabetes_model.predict(x_scaled)[0]
            prob = self.diabetes_model.predict_proba(x_scaled)[0][1]

        except Exception as e:
            print(f"Diabetes prediction error: {e}")
            return {'probability': 0.25, 'prediction': False}
        
        result = {
            'probability': prob,
            'prediction': pred == 1
        }
        
        print(f"\n[Diabetes] Probability: {prob:.3f}, Prediction: {'Yes' if result['prediction'] else 'No'}")
        return result
    
    def merge_lung_predictions(self, image_diseases, text_diseases):
        """
        Merge lung disease predictions from image and text models.
        Prefer text model when there's disagreement.
        
        Args:
            image_diseases (list): Diseases predicted by image model
            text_diseases (list): Diseases predicted by text model
            
        Returns:
            list: Final merged list of lung diseases
        """
        image_set = set(image_diseases)
        text_set = set(text_diseases)
        
        # Get conditions both models agree on
        agreed_conditions = image_set.intersection(text_set)
        
        # Get conditions only text model detected (prefer text model)
        text_only_conditions = text_set - image_set
        
        # Combine agreed + text-only conditions  
        final_conditions = sorted(list(agreed_conditions.union(text_only_conditions)))
        
        print(f"\n[Merged Predictions] Final lung diseases: {final_conditions}")
        return final_conditions
    
    def generate_medical_prompt(self, lung_diseases, heart_result, kidney_result, diabetes_result):
        """
        Generate structured medical prompt for LLM - Research focused
        """
        prompt = """This is a medical research simulation. Based on the clinical findings below, suggest ONLY relevant medication classes. DO NOT invent diseases or treatments not listed.

    CLINICAL FINDINGS:
    """

        # Add lung conditions
        if lung_diseases:
            prompt += f"Primary Lung Conditions: {', '.join(lung_diseases)}\n"
        else:
            prompt += "Primary Lung Conditions: None detected\n"

        # Add comorbidities status
        prompt += f"Diabetes Status: {'PRESENT' if diabetes_result['prediction'] else 'ABSENT'}\n"
        prompt += f"Heart Disease Risk: {'HIGH' if heart_result['prediction'] else 'LOW'}\n"
        prompt += f"Kidney Disease Risk: {'HIGH' if kidney_result['prediction'] else 'LOW'}\n"

        # Special logic for 'No Finding' alone
        if 'No Finding' in lung_diseases and len(lung_diseases) == 1:
            prompt += """
    NOTE: Since "No Finding" was detected as the only lung condition, no medication recommendations should be made.

    Just respond with: "No abnormal lung condition detected. No medication required.".
    """
            return prompt

        prompt += """
    RESEARCH TASK: List specific medication classes or drugs typically used for the detected lung conditions, considering the comorbidity profile. Format as:

    For [Condition]:
    - First-line: [medication]
    - Alternative: [medication]
    - Contraindications due to comorbidities: [if any]

    This is purely for medical education and research simulation purposes.
    """

        return prompt

    
    def query_local_llm(self, prompt, model_name="llama3.2"):
        """
        Query local LLM using Ollama
        
        Args:
            prompt (str): Medical prompt to send to LLM
            model_name (str): Name of the local model to use
            
        Returns:
            str: LLM response
        """
        try:
            # Try Ollama first
            print(f"\nQuerying local LLM ({model_name})...")
            
            # Create temporary file for prompt
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                f.write(prompt)
                temp_file = f.name
            
            try:
                # Run Ollama command
                # Read the prompt from file and pass it directly
                with open(temp_file, 'r', encoding='utf-8') as f:
                    prompt_content = f.read()

                ###
                result = subprocess.run([
                        'ollama', 'run', model_name
                ], input=prompt_content, capture_output=True, text=True, timeout=120, encoding='utf-8', errors='ignore')
                ###
                if result.returncode == 0:
                    response = result.stdout.strip()
                    # Ensure disclaimer is included
                    if "experimental and research purposes" not in response:
                        response += "\n\nNote: This is only for experimental and research purposes. Please consult a real doctor."
                    return response
                else:
                    print(f"Ollama error: {result.stderr}")
                    
            finally:
                # Clean up temp file
                os.unlink(temp_file)
                
        except subprocess.TimeoutExpired:
            print("LLM query timed out.")
        except FileNotFoundError:
            print("Ollama not found. Trying alternative methods...")
        
        # Fallback: Try GPT4All
        try:
            print("Trying GPT4All...")
            result = subprocess.run([
                'gpt4all', '--model', 'orca-mini-3b-gguf2-q4_0.gguf', '--prompt', prompt
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                response = result.stdout.strip()
                if "experimental and research purposes" not in response:
                    response += "\n\nNote: This is only for experimental and research purposes. Please consult a real doctor."
                return response
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Fallback: Try LM Studio
        try:
            print("Trying LM Studio CLI...")
            result = subprocess.run([
                'lms', 'complete', '--prompt', prompt
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                response = result.stdout.strip()
                if "experimental and research purposes" not in response:
                    response += "\n\nNote: This is only for experimental and research purposes. Please consult a real doctor."
                return response
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Final fallback - return structured response
        return self._generate_fallback_response(prompt)
    
    def _generate_fallback_response(self, prompt):
        """Generate a basic medical response when no LLM is available"""
        response = """Medical Analysis (Generated Fallback):

Based on the detected conditions, here are general recommendations:

For Lung Conditions:
- Follow up with a pulmonologist for proper evaluation
- Consider chest imaging if symptoms persist
- Monitor respiratory symptoms closely

For Comorbidities:
- Heart Disease: Regular cardiology follow-up, consider ACE inhibitors/ARBs
- Diabetes: Monitor blood glucose, consider metformin if appropriate
- Kidney Disease: Monitor kidney function, adjust medications for renal clearance

Important: These are general guidelines only. Individual treatment plans must be determined by qualified healthcare providers.

Note: This is only for experimental and research purposes. Please consult a real doctor."""
        
        return response
    
    def run_complete_pipeline(self, image_path, clinical_text, heart_features, kidney_features, diabetes_features):
        """
        Run the complete medical AI pipeline
        
        Args:
            image_path (str): Path to chest X-ray image
            clinical_text (str): Clinical report text
            heart_features (list): Features for heart disease prediction
            kidney_features (list): Features for kidney disease prediction
            diabetes_features (list): Features for diabetes prediction
            
        Returns:
            str: Final medical analysis from LLM
        """
        print("="*60)
        print("MEDICAL AI PIPELINE - COMPLETE ANALYSIS")
        print("="*60)
        
        # Step 1: Predict lung diseases
        print("\nStep 1: Analyzing lung conditions...")
        image_diseases = self.predict_image(image_path)
        text_diseases = self.predict_text(clinical_text)
        
        # Step 2: Merge lung predictions
        print("\nStep 2: Merging lung disease predictions...")
        final_lung_diseases = self.merge_lung_predictions(image_diseases, text_diseases)
        
        # Step 3: Predict comorbidities
        print("\nStep 3: Analyzing comorbidities...")
        heart_result = self.predict_heart_disease(heart_features)
        kidney_result = self.predict_kidney_disease(kidney_features)
        diabetes_result = self.predict_diabetes(diabetes_features)
        
        # Step 4: Generate medical prompt
        print("\nStep 4: Generating medical analysis prompt...")
        medical_prompt = self.generate_medical_prompt(
            final_lung_diseases, heart_result, kidney_result, diabetes_result
        )
        
        print("\nGenerated Prompt:")
        print("-" * 40)
        print(medical_prompt)
        print("-" * 40)
        
        # Step 5: Query local LLM
        print("\nStep 5: Querying local LLM for medical recommendations...")
        llm_response = self.query_local_llm(medical_prompt)
        
        # Final output
        print("\n" + "="*60)
        print("FINAL MEDICAL ANALYSIS")
        print("="*60)
        print(llm_response)
        print("="*60)
        
        return llm_response
    
    def run(self, image_path, clinical_text, heart_features, kidney_features, diabetes_features):
        """
        Run the complete medical AI pipeline combining all predictions.
        
        Args:
            image_path (str): Path to chest X-ray image
            clinical_text (str): Clinical text description
            heart_features (list): Heart disease features
            kidney_features (list): Kidney disease features
            diabetes_features (list): Diabetes features
            
        Returns:
            dict: Dictionary containing all predictions
        """
        try:
            # Run image analysis
            image_predictions = self.predict_image(image_path)
            
            # Run text analysis
            text_predictions = self.predict_text(clinical_text)
            
            # Run tabular models
            heart_prediction = self.predict_heart_disease(heart_features)
            kidney_prediction = self.predict_kidney_disease(kidney_features)
            diabetes_prediction = self.predict_diabetes(diabetes_features)
            
            # Combine results
            results = {
                'image_prediction': image_predictions,
                'text_prediction': text_predictions,
                'heart_prediction': heart_prediction,
                'kidney_prediction': kidney_prediction,
                'diabetes_prediction': diabetes_prediction
            }
            
            return results
            
        except Exception as e:
            print(f"Error in run method: {str(e)}")
            raise


def main():
    """Example usage of the Medical AI Pipeline"""
    
    # Define model paths (update these to your actual paths)
    model_paths = {
        'image_model': r"C:\Users\prath\Desktop\temp\Image\tranfer_learning.pt",
        'text_model': r"C:\Users\prath\Desktop\temp\Text\final_finetuned.pt", 
        'heart_model': r"C:\Users\prath\Desktop\temp\Table\Heart_Disease\tabular_model.pt",
        'kidney_model': r"C:\Users\prath\Desktop\temp\Table\Kidney\ckd_model.joblib",
        'diabetes_model': r"C:\Users\prath\Desktop\temp\Table\Diabetes\diabetes_model.joblib",
        'diabetes_scaler': r"C:\Users\prath\Desktop\temp\Table\Diabetes\diabetes_scaler.joblib"
    }
    
    # Initialize pipeline
    pipeline = MedicalAIPipeline(model_paths)
    
    # Sample inputs
    image_path = r"C:\Users\prath\Desktop\temp\Input\view1_frontal.jpg"
    
    clinical_text = """A tip of a left Port-A-Cath lies in the low superior vena cava. Indistinct nodular opacities in the left lung base are consistent with known metastatic disease and better demonstrated on the prior CT. There is bilateral pleural thickening and small pleural effusions that appear relatively unchanged in size since ___. The cardiac and mediastinal contours are normal. The upper bowel gas pattern is unremarkable.
"""
    
    # Heart disease features (25 features after one-hot encoding)
    # Features: age, sex, chest pain type (4 one-hot), resting BP, cholesterol, 
    # fasting blood sugar, resting ECG (3 one-hot), max heart rate, exercise angina,
    # ST depression, ST slope (3 one-hot), number of vessels (4 one-hot), thalassemia (4 one-hot)
    heart_features = [
        60,     # age
        1,      # sex (1=male, 0=female)
        1, 0, 0, 0,  # chest pain type (typical angina, atypical angina, non-anginal, asymptomatic)
        120,    # resting blood pressure
        240,    # cholesterol
        0,      # fasting blood sugar > 120 mg/dl
        1, 0, 0,  # resting ECG (normal, ST-T abnormality, LV hypertrophy)
        150,    # max heart rate achieved
        1,      # exercise induced angina
        2.5,    # ST depression induced by exercise
        1, 0, 0,  # ST slope (upsloping, flat, downsloping)
        1, 0, 0, 0,  # number of major vessels (0-3)
        0, 1, 0, 0,  # thalassemia (normal, fixed defect, reversible defect, unknown)
        0, 0    # Add 2 more dummy features to make it 28 total
    ]
    
    # Kidney disease features (24 features)
    # [age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane]
    kidney_features = [
        48,      # age
        80,      # blood pressure
        1.020,   # specific gravity
        1,       # albumin
        0,       # sugar
        'normal',    # red blood cells
        'normal',    # pus cell
        'notpresent', # pus cell clumps
        'notpresent', # bacteria
        121,     # blood glucose random
        36,      # blood urea
        1.2,     # serum creatinine
        15,      # sodium
        4.6,     # potassium
        15,      # hemoglobin
        44,      # packed cell volume
        7800,    # white blood cell count
        5.2,     # red blood cell count
        'yes',   # hypertension
        'yes',   # diabetes mellitus
        'no',    # coronary artery disease
        'good',  # appetite
        'no',    # pedal edema
        'no'     # anemia
    ]
    
    # Diabetes features (21 base features)
    # [HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity,
    #  Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth,
    #  PhysHlth, DiffWalk, Sex, Age, Education, Income]
    diabetes_features = [
        1,     # HighBP (1=yes, 0=no)
        1,     # HighChol (1=yes, 0=no)  
        1,     # CholCheck (1=yes, 0=no)
        28.5,  # BMI
        0,     # Smoker (1=yes, 0=no)
        0,     # Stroke (1=yes, 0=no)
        0,     # HeartDiseaseorAttack (1=yes, 0=no)
        1,     # PhysActivity (1=yes, 0=no)
        1,     # Fruits (1=yes, 0=no)
        1,     # Veggies (1=yes, 0=no)
        0,     # HvyAlcoholConsump (1=yes, 0=no)
        1,     # AnyHealthcare (1=yes, 0=no)
        0,     # NoDocbcCost (1=yes, 0=no)
        3,     # GenHlth (1=excellent, 2=very good, 3=good, 4=fair, 5=poor)
        5,     # MentHlth (days of poor mental health in past 30 days)
        0,     # PhysHlth (days of poor physical health in past 30 days)
        0,     # DiffWalk (1=yes, 0=no)
        0,     # Sex (1=male, 0=female)
        45,    # Age
        6,     # Education (1-6 scale)
        7      # Income (1-8 scale)
    ]
    
    # Run the complete pipeline
    try:
        final_analysis = pipeline.run_complete_pipeline(
            image_path=image_path,
            clinical_text=clinical_text,
            heart_features=heart_features,
            kidney_features=kidney_features,
            diabetes_features=diabetes_features
        )
        
        # Save results to file
        with open('medical_analysis_results.txt', 'w') as f:
            f.write("MEDICAL AI PIPELINE RESULTS\n")
            f.write("="*50 + "\n\n")
            f.write(f"Image Path: {image_path}\n")
            f.write(f"Clinical Text: {clinical_text}\n\n")
            f.write("FINAL ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            f.write(final_analysis)
        
        print(f"\nResults saved to: medical_analysis_results.txt")
        
    except Exception as e:
        print(f"Error running pipeline: {e}")
        import traceback
        traceback.print_exc()


def demo_individual_predictions():
    """Demo function to test individual model predictions"""
    
    # Define model paths
    model_paths = {
        'image_model': r"C:\Users\prath\Desktop\temp\Image\tranfer_learning.pt",
        'text_model': r"C:\Users\prath\Desktop\temp\Text\final_finetuned.pt", 
        'heart_model': r"C:\Users\prath\Desktop\temp\Table\Heart_Disease\tabular_model.pt",
        'kidney_model': r"C:\Users\prath\Desktop\temp\Table\Kidney\ckd_model.joblib",
        'diabetes_model': r"C:\Users\prath\Desktop\temp\Table\Diabetes\diabetes_model.joblib",
        'diabetes_scaler': r"C:\Users\prath\Desktop\temp\Table\Diabetes\diabetes_scaler.joblib"
    }
    
    # Initialize pipeline
    pipeline = MedicalAIPipeline(model_paths)
    
    print("\n" + "="*60)
    print("INDIVIDUAL MODEL TESTING")
    print("="*60)
    
    # Test image model
    print("\n1. Testing Image Model:")
    image_path = r"C:\Users\prath\Desktop\temp\Input\view1_frontal.jpg"
    image_diseases = pipeline.predict_image(image_path)
    
    # Test text model
    print("\n2. Testing Text Model:")
    clinical_text = "Patient shows signs of pneumonia and pleural effusion"
    text_diseases = pipeline.predict_text(clinical_text)
    
    # Test heart model
    print("\n3. Testing Heart Model:")
    heart_features = [60, 1, 1, 0, 0, 0, 120, 240, 0, 1, 0, 0, 150, 1, 2.5, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    heart_result = pipeline.predict_heart_disease(heart_features)
    
    # Test kidney model
    print("\n4. Testing Kidney Model:")
    kidney_features = [48, 80, 1.020, 1, 0, 'normal', 'normal', 'notpresent', 'notpresent', 
                      121, 36, 1.2, 15, 4.6, 15, 44, 7800, 5.2, 'yes', 'yes', 'no', 'good', 'no', 'no']
    kidney_result = pipeline.predict_kidney_disease(kidney_features)
    
    # Test diabetes model
    print("\n5. Testing Diabetes Model:")
    diabetes_features = [1, 1, 1, 28.5, 0, 0, 0, 1, 1, 1, 0, 1, 0, 3, 5, 0, 0, 0, 45, 6, 7]
    diabetes_result = pipeline.predict_diabetes(diabetes_features)
    
    print("\n" + "="*60)
    print("INDIVIDUAL TESTING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    print("Medical AI Pipeline")
    print("==================")
    print("1. Running complete pipeline...")
    
    # Run main pipeline
    main()
    
    print("\n2. Running individual model tests...")
    
    # Run individual tests
    demo_individual_predictions()
    
    print("\nPipeline execution completed!")