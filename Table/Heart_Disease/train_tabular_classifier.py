import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import random

# Define the classes
COMMON_CLASSES = [
    'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
    'Fracture', 'Support Devices', 'No Finding'
]

def generate_synthetic_data(num_samples=1000):
    """Generate synthetic patient data with realistic distributions."""
    data = {
        # Demographics
        'age': np.random.normal(60, 15, num_samples).clip(18, 95),
        'gender': np.random.choice(['M', 'F'], num_samples),
        'bmi': np.random.normal(26, 5, num_samples).clip(15, 45),
        'smoking_status': np.random.choice(['Never', 'Former', 'Current'], num_samples),
        
        # Vital Signs
        'heart_rate': np.random.normal(80, 15, num_samples).clip(40, 150),
        'systolic_bp': np.random.normal(125, 20, num_samples).clip(80, 200),
        'diastolic_bp': np.random.normal(80, 15, num_samples).clip(40, 120),
        'respiratory_rate': np.random.normal(16, 4, num_samples).clip(8, 40),
        'temperature': np.random.normal(37, 0.5, num_samples).clip(35, 40),
        'spo2': np.random.normal(96, 3, num_samples).clip(80, 100),
        
        # Lab Values
        'wbc_count': np.random.normal(8, 3, num_samples).clip(2, 20),
        'crp': np.random.exponential(30, num_samples).clip(0, 300),
        'd_dimer': np.random.exponential(300, num_samples).clip(0, 3000),
        'ldh': np.random.normal(250, 100, num_samples).clip(100, 1000),
        'platelet_count': np.random.normal(250, 100, num_samples).clip(50, 600),
        
        # Clinical History (Binary)
        'previous_lung_disease': np.random.choice([0, 1], num_samples),
        'previous_heart_disease': np.random.choice([0, 1], num_samples),
        'previous_surgery': np.random.choice([0, 1], num_samples),
        'chronic_respiratory_condition': np.random.choice([0, 1], num_samples),
        'recent_trauma': np.random.choice([0, 1], num_samples)
    }
    
    # Generate synthetic labels (multi-label)
    labels = np.zeros((num_samples, len(COMMON_CLASSES)))
    for i in range(num_samples):
        # Each patient can have 0-3 conditions
        num_conditions = np.random.choice([0, 1, 2, 3])
        if num_conditions == 0:
            labels[i, COMMON_CLASSES.index('No Finding')] = 1
        else:
            # Select random conditions (excluding 'No Finding')
            possible_conditions = list(range(len(COMMON_CLASSES)-1))  # Exclude 'No Finding'
            selected_conditions = np.random.choice(possible_conditions, num_conditions, replace=False)
            labels[i, selected_conditions] = 1
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add labels
    for i, class_name in enumerate(COMMON_CLASSES):
        df[f'label_{class_name}'] = labels[:, i]
    
    return df

class TabularDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class TabularClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64]):
        super(TabularClassifier, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, len(COMMON_CLASSES)))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def preprocess_data(df):
    """Preprocess the data for training."""
    # Separate features and labels
    label_cols = [f'label_{class_name}' for class_name in COMMON_CLASSES]
    feature_cols = [col for col in df.columns if col not in label_cols]
    
    # Convert categorical variables
    df = pd.get_dummies(df, columns=['gender', 'smoking_status'])
    
    # Standardize numerical features
    scaler = StandardScaler()
    numerical_cols = ['age', 'bmi', 'heart_rate', 'systolic_bp', 'diastolic_bp',
                     'respiratory_rate', 'temperature', 'spo2', 'wbc_count',
                     'crp', 'd_dimer', 'ldh', 'platelet_count']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Get features and labels
    features = df.drop(label_cols, axis=1).values
    labels = df[label_cols].values
    
    return features, labels

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    """Train the model."""
    model = model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_features)
                val_loss += criterion(outputs, batch_labels).item()
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_tabular_model.pt')

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    df = generate_synthetic_data(num_samples=10000)
    
    # Preprocess data
    print("Preprocessing data...")
    features, labels = preprocess_data(df)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize model
    input_size = features.shape[1]
    model = TabularClassifier(input_size)
    
    # Define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    print("Training model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)
    
    print("Training complete! Model saved as 'best_tabular_model.pt'")

if __name__ == "__main__":
    main() 