import torch
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from train_tabular_classifier import (
    TabularClassifier, generate_synthetic_data, preprocess_data, COMMON_CLASSES
)

def evaluate_model(model, test_features, test_labels, device='cuda'):
    """Evaluate the model on test data."""
    model.eval()
    with torch.no_grad():
        # Convert to tensors
        test_features = torch.FloatTensor(test_features).to(device)
        test_labels = torch.FloatTensor(test_labels).to(device)
        
        # Get predictions
        outputs = model(test_features)
        predictions = torch.sigmoid(outputs).cpu().numpy()
        
        # Convert to binary predictions
        binary_predictions = (predictions > 0.5).astype(int)
        
        # Calculate metrics
        print("\nClassification Report:")
        print(classification_report(
            test_labels.cpu().numpy(), 
            binary_predictions,
            target_names=COMMON_CLASSES
        ))
        
        # Calculate ROC AUC for each class
        print("\nROC AUC Scores:")
        for i, class_name in enumerate(COMMON_CLASSES):
            auc = roc_auc_score(test_labels.cpu().numpy()[:, i], predictions[:, i])
            print(f"{class_name}: {auc:.4f}")

def main():
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate test data
    print("Generating test data...")
    test_df = generate_synthetic_data(num_samples=2000)
    
    # Preprocess data
    print("Preprocessing data...")
    test_features, test_labels = preprocess_data(test_df)
    
    # Load model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TabularClassifier(input_size=test_features.shape[1])
    model.load_state_dict(torch.load('best_tabular_model.pt'))
    model = model.to(device)
    
    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, test_features, test_labels, device)

if __name__ == "__main__":
    main() 