import torch
import torchxrayvision as xrv
import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
from tqdm import tqdm

LABEL_COLUMNS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices'
]

class ChestXRayDataset(Dataset):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        
        # Read the CSV file containing image paths and labels
        csv_path = os.path.join(base_dir, 'train.csv')
        self.df = pd.read_csv(csv_path)
        
        # Process labels (convert -1 to 0 as per CheXpert paper recommendation)
        self.labels = self.df[LABEL_COLUMNS].fillna(0).replace(-1, 1).values
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        print(f"Loaded dataset with {len(self.df)} images")

    def __len__(self):
        return len(self.df)    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Extract the filename from the path and construct the correct local path
        img_filename = os.path.basename(row['Path'])
        img_path = os.path.join(self.base_dir, 'train', img_filename)
        
        try:
            # Load and preprocess image
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            
            # Get labels
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            
            return image, label
        except FileNotFoundError:
            print(f"Warning: Image not found at {img_path}")
            # Return a blank image and zeros for labels as fallback
            image = torch.zeros((1, 224, 224))
            label = torch.zeros(len(LABEL_COLUMNS), dtype=torch.float32)
            return image, label

def evaluate_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained model
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model = model.to(device)
    model.eval()

    # Load dataset
    data_dir = r"C:\Users\prath\Desktop\Pra_Projects\Multi Modal Health Insights Platform\data\raw\images\archive (2)"
    dataset = ChestXRayDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # Evaluation
    all_preds = []
    all_labels = []

    print("Starting evaluation...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()  # Binary classification threshold
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)    # Calculate metrics per class
    results = {}
    print("\nPer-class Results:")
    for i, label_name in enumerate(LABEL_COLUMNS):
        class_preds = all_preds[:, i]
        class_labels = all_labels[:, i]
        
        f1 = f1_score(class_labels, class_preds, average='binary')
        precision = precision_score(class_labels, class_preds, average='binary')
        recall = recall_score(class_labels, class_preds, average='binary')
        
        print(f"\n{label_name}:")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
        results[label_name] = {
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall)
        }

    # Calculate macro averages
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    macro_precision = precision_score(all_labels, all_preds, average='macro')
    macro_recall = recall_score(all_labels, all_preds, average='macro')

    print("\nOverall Macro Averages:")
    print(f"F1 Score (Macro): {macro_f1:.4f}")
    print(f"Precision (Macro): {macro_precision:.4f}")
    print(f"Recall (Macro): {macro_recall:.4f}")

    results['macro_averages'] = {
        'f1': float(macro_f1),
        'precision': float(macro_precision),
        'recall': float(macro_recall)
    }
    
    with open('torchxrayvision_evaluation_results.json', 'w') as f:
        import json
        json.dump(results, f, indent=4)

    return results

if __name__ == "__main__":
    evaluate_model()
