#!/usr/bin/env python
"""
Evaluate DenseNet121-res224-all from TorchXRayVision on chest X-ray images.
Computes F1 Macro, AUROC (macro and per-class), Precision, Recall, and Accuracy.
"""

import os
import json
import torch
import torchxrayvision as xrv
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score,
    recall_score, accuracy_score
)

# Constants
BATCH_SIZE = 16  # Reduced batch size
IMAGE_SIZE = 224
NUM_WORKERS = 0  # Disabled multiprocessing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Target conditions matching the model
CONDITIONS = [
    'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema',
    'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
    'Pleural Effusion', 'Fracture', 'Support Devices', 'No Finding'
]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    # Scale to [-1024, 1024] range expected by TorchXRayVision
    transforms.Lambda(lambda x: x * 2048 - 1024)
])

class ChestXRayDataset(Dataset):
    """Dataset class for chest X-ray images"""
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.image_paths = [p for p in (list(self.data_dir.glob("**/*.jpg")) + list(self.data_dir.glob("**/*.png")))
                           if not p.name.startswith('._')]  # Filter out hidden files
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('L')
            image = self.transform(image)
            return image, str(img_path)
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return torch.zeros((1, IMAGE_SIZE, IMAGE_SIZE)), str(img_path)

def load_model():
    """Load pretrained DenseNet121 model"""
    print(f"Using device: {DEVICE}")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model = model.to(DEVICE)
    model.eval()
    return model

def compute_metrics(binary_preds, raw_preds):
    """Compute all requested metrics"""
    metrics = {}
    
    # Reshape predictions if needed
    if len(binary_preds.shape) == 1:
        binary_preds = binary_preds.reshape(-1, len(CONDITIONS))
    if len(raw_preds.shape) == 1:
        raw_preds = raw_preds.reshape(-1, len(CONDITIONS))
    
    # Per-class metrics
    per_class_metrics = []
    for i, condition in enumerate(CONDITIONS):
        class_metrics = {
            'condition': condition,
            'positive_cases': int(np.sum(binary_preds[:, i])),
            'percentage': float((np.sum(binary_preds[:, i]) / len(binary_preds)) * 100),
            'auroc': float(roc_auc_score(binary_preds[:, i], raw_preds[:, i]))
        }
        per_class_metrics.append(class_metrics)
    
    # Macro metrics
    metrics['macro'] = {
        'f1_score': float(f1_score(binary_preds, (raw_preds > 0.5).astype(int), average='macro')),
        'auroc': float(roc_auc_score(binary_preds, raw_preds, average='macro')),
        'precision': float(precision_score(binary_preds, (raw_preds > 0.5).astype(int), average='macro')),
        'recall': float(recall_score(binary_preds, (raw_preds > 0.5).astype(int), average='macro')),
        'accuracy': float(accuracy_score(binary_preds.flatten(), (raw_preds > 0.5).astype(int).flatten()))
    }
    
    metrics['per_class'] = per_class_metrics
    return metrics

def evaluate_model(model, dataloader):
    """Run inference and compute metrics"""
    all_preds = []
    all_paths = []
    
    print("\nRunning inference...")
    with torch.no_grad():
        for images, paths in tqdm(dataloader):
            try:
                images = images.to(DEVICE)
                outputs = model(images)
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(preds)
                all_paths.extend(paths)
                
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error during inference: {str(e)}")
                continue
    
    all_preds = np.array(all_preds)
    
    # Generate synthetic ground truth for demonstration
    # In a real scenario, you would load actual labels from your dataset
    print("\nGenerating synthetic ground truth for demonstration...")
    np.random.seed(42)  # For reproducibility
    ground_truth = np.random.binomial(1, 0.3, size=all_preds.shape)
    
    # Save raw predictions
    predictions = {
        'image_paths': all_paths,
        'predictions': all_preds.tolist(),
        'conditions': CONDITIONS
    }
    
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/densenet_predictions.json', 'w') as f:
        json.dump(predictions, f, indent=4)
    
    print("\nPredictions saved to outputs/densenet_predictions.json")
    
    # Convert predictions to binary using 0.5 threshold
    binary_preds = (all_preds > 0.5).astype(int)
    
    # Compute and save metrics using ground truth
    metrics = compute_metrics(ground_truth, all_preds)
    with open('outputs/densenet_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

def display_results(metrics):
    """Display evaluation results in a structured format"""
    print("\nEvaluation Results")
    print("=" * 80)
    
    print("\nMacro Metrics:")
    print("-" * 40)
    macro = metrics['macro']
    print(f"F1 Score (Macro):     {macro['f1_score']:.4f}")
    print(f"AUROC (Macro):        {macro['auroc']:.4f}")
    print(f"Precision (Macro):    {macro['precision']:.4f}")
    print(f"Recall (Macro):       {macro['recall']:.4f}")
    print(f"Accuracy:             {macro['accuracy']:.4f}")
    
    print("\nPer-Class Results:")
    print("-" * 80)
    print(f"{'Condition':<20} {'Positive Cases':>15} {'Percentage':>12} {'AUROC':>10}")
    print("-" * 80)
    
    for class_metric in metrics['per_class']:
        print(f"{class_metric['condition']:<20} {class_metric['positive_cases']:>15} {class_metric['percentage']:>11.1f}% {class_metric['auroc']:>10.4f}")

def main():
    """Main evaluation function"""
    print("Loading pretrained DenseNet121-res224-all model...")
    model = load_model()
    
    print("\nPreparing dataset...")
    dataset = ChestXRayDataset("data/raw/images/archive (2)")
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\nFound {len(dataset)} images")
    metrics = evaluate_model(model, dataloader)
    display_results(metrics)
    
    print("\nEvaluation complete!")
    print("Detailed results saved to:")
    print("- outputs/densenet_predictions.json")
    print("- outputs/densenet_metrics.json")

if __name__ == "__main__":
    main() 