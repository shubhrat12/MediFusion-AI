import torch
import torchxrayvision as xrv
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import logging
import pandas as pd
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score

# Set up logging
logging.basicConfig(
    filename=f'xray_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ChestXRayDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = []
        self.labels = []
        self.skipped_files = []
        
        # Define pathology mapping to match CSV columns
        self.pathology_mapping = {
            'Atelectasis': 'Atelectasis',
            'Cardiomegaly': 'Cardiomegaly',
            'Consolidation': 'Consolidation',
            'Edema': 'Edema',
            'Pleural Effusion': 'Pleural Effusion',
            'Pneumonia': 'Pneumonia',
            'Pneumothorax': 'Pneumothorax',
            'Lung Opacity': 'Lung Opacity',
            'Lung Lesion': 'Lung Lesion',
            'Enlarged Cardiomediastinum': 'Enlarged Cardiomediastinum',
            'Fracture': 'Fracture',
            'Pleural Other': 'Pleural Other',
            'Support Devices': 'Support Devices'
        }
        
        # Read the CSV file
        csv_path = os.path.join(data_dir, 'train.csv')
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
            logging.info(f"Loaded CSV with {len(self.df)} entries")
            
            # Create label matrix
            self.label_cols = list(self.pathology_mapping.keys())
            
            # Process images from CSV paths
            for _, row in self.df.iterrows():
                img_path = row['Path'].replace('CheXpert-v1.0-small/train/', '')
                full_path = os.path.join(data_dir, 'train', img_path)
                try:
                    # Verify image can be opened
                    with Image.open(full_path) as img:
                        if img.mode not in ['L', 'RGB']:
                            self.skipped_files.append((full_path, f"Unsupported image mode: {img.mode}"))
                            continue
                    self.image_files.append(full_path)
                    # Get labels for this image
                    label_row = [float(row[col]) if pd.notna(row[col]) else 0.0 for col in self.label_cols]
                    # Replace -1 (uncertain) with 0 for this evaluation
                    label_row = [1.0 if x == 1.0 else 0.0 for x in label_row]
                    self.labels.append(label_row)
                except Exception as e:
                    self.skipped_files.append((full_path, str(e)))
                    logging.warning(f"Skipping corrupted image {full_path}: {str(e)}")
        else:
            logging.warning(f"No CSV found at {csv_path}, will only process images")
            self.df = None
            self.label_matrix = None
        
        logging.info(f"Found {len(self.image_files)} valid images with labels")
        if self.skipped_files:
            logging.warning(f"Skipped {len(self.skipped_files)} problematic images")
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])  # ImageNet stats for single channel
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        try:
            with Image.open(img_path) as image:
                # Convert to grayscale if RGB
                if image.mode == 'RGB':
                    image = image.convert('L')
                image = self.transform(image)
                return image.repeat(1, 1, 1), labels, img_path  # Return image, labels, and path
        except Exception as e:
            logging.error(f"Error loading {img_path}: {str(e)}")
            return torch.zeros((1, 224, 224)), labels, img_path

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    try:
        # Load pretrained model
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        model = model.to(device)
        model.eval()
        logging.info("Model loaded successfully")

        # Load dataset
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               "data", "raw", "images", "archive (2)")
        dataset = ChestXRayDataset(data_dir)
        dataloader = DataLoader(dataset, 
                              batch_size=32, 
                              shuffle=False, 
                              num_workers=2,
                              pin_memory=True)

        logging.info(f"Processing {len(dataset)} images...")
        
        # Initialize tracking variables
        all_predictions = []
        all_labels = []
        pathology_scores = {p: [] for p in model.pathologies}
        pathology_labels = {p: [] for p in model.pathologies}
        pathology_predictions = {p: [] for p in model.pathologies}
        pathology_confidence = {p: [] for p in model.pathologies}
        failed_images = []
        processed_count = 0
        batch_times = []

        confidence_threshold = 0.5  # Lowered from 0.8 for more reasonable tracking

        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc="Processing batches"):
                batch, labels, paths = batch_data
                batch_start = datetime.now()
                
                try:
                    batch = batch.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    output = model(batch)
                    
                    # Convert outputs to binary predictions
                    predictions = (output > 0.5).float()
                    
                    # Store predictions and labels
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    # Process outputs per pathology
                    for i, (pred, label, path) in enumerate(zip(predictions, labels, paths)):
                        for j, (pathology, score) in enumerate(zip(model.pathologies, output[i])):
                            if j < len(dataset.label_cols):  # Only process pathologies we have labels for
                                score = score.item()
                                label_val = label[j].item()
                                pred_val = pred[j].item()
                                
                                pathology_scores[pathology].append(score)
                                pathology_labels[pathology].append(label_val)
                                pathology_predictions[pathology].append(pred_val)
                                
                                # Track high confidence predictions
                                if abs(score) > confidence_threshold:
                                    pathology_confidence[pathology].append({
                                        'path': path,
                                        'score': score,
                                        'true_label': label_val,
                                        'predicted': pred_val
                                    })
                        
                        processed_count += 1
                        
                except Exception as e:
                    failed_images.extend(paths)
                    logging.error(f"Error processing batch: {str(e)}")
                    continue
                
                batch_end = datetime.now()
                batch_times.append((batch_end - batch_start).total_seconds())

        # Calculate metrics for each pathology
        results = {
            "pathology_metrics": {},
            "performance_metrics": {
                "total_images": len(dataset),
                "processed_images": processed_count,
                "failed_images": len(failed_images),
                "avg_batch_time": np.mean(batch_times),
            }
        }

        # Calculate and log results
        print("\nEvaluation Results by Pathology:")
        print("-" * 50)
        
        all_accuracies = []
        all_f1_scores = []
        
        for pathology in model.pathologies:
            if pathology in dataset.pathology_mapping:
                true_labels = np.array(pathology_labels[pathology])
                predictions = np.array(pathology_predictions[pathology])
                
                if len(true_labels) > 0:
                    accuracy = accuracy_score(true_labels, predictions)
                    f1 = f1_score(true_labels, predictions, zero_division=0)
                    high_conf_correct = sum(1 for x in pathology_confidence[pathology] 
                                         if x['true_label'] == x['predicted'])
                    high_conf_total = len(pathology_confidence[pathology])
                    
                    all_accuracies.append(accuracy)
                    all_f1_scores.append(f1)
                    
                    results["pathology_metrics"][pathology] = {
                        "accuracy": float(accuracy),
                        "f1_score": float(f1),
                        "high_confidence_total": high_conf_total,
                        "high_confidence_correct": high_conf_correct,
                    }
                    
                    print(f"\n{pathology}:")
                    print(f"  Accuracy: {accuracy:.3f}")
                    print(f"  F1 Score: {f1:.3f}")
                    print(f"  High Confidence Predictions: {high_conf_total} (Correct: {high_conf_correct})")

        # Calculate overall metrics
        results["overall_metrics"] = {
            "accuracy": float(np.mean(all_accuracies)),
            "macro_f1_score": float(np.mean(all_f1_scores))
        }

        print("\nOverall Metrics:")
        print(f"Average Accuracy: {results['overall_metrics']['accuracy']:.4f}")
        print(f"Average Macro F1 Score: {results['overall_metrics']['macro_f1_score']:.4f}")

        if failed_images:
            print(f"\nFailed to process {len(failed_images)} images")
            logging.warning("Failed images:")
            for img in failed_images:
                logging.warning(f"  {img}")

        # Save results to JSON
        results_file = f'evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to {results_file}")

        return results

    except Exception as e:
        logging.error(f"Critical error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    evaluate_model()
