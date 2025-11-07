import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle
from pathlib import Path

from model import create_model
from utils import DataPreprocessor, NetworkTrafficDataset, get_class_weights, calculate_metrics, save_model
from config import ModelConfig, TrainingConfig, DatasetConfig, PROJECT_ROOT, MODEL_DIR, DATA_DIR

class Trainer:
    """Training orchestrator for Network IDS model"""
    
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": []
        }
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        for X, y in progress_bar:
            X, y = X.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, criterion):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                loss = criterion(outputs, y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)
    
    def fit(self, train_loader, val_loader, epochs=50, learning_rate=0.001):
        """Train model with early stopping"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
        
        print(f"\n{'='*60}")
        print(f"Training Configuration")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Device: {self.device}")
        print(f"Batch Size: {len(train_loader.dataset) // len(train_loader)}")
        print(f"{'='*60}\n")
        
        epoch_bar = tqdm(range(epochs), desc="Epochs")
        
        for epoch in epoch_bar:
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc, val_preds, val_labels = self.validate(val_loader, criterion)
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["val_accuracy"].append(val_acc)
            
            epoch_bar.set_postfix({
                "train_loss": f"{train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "train_acc": f"{train_acc:.2f}%",
                "val_acc": f"{val_acc:.2f}%"
            })
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                save_path = MODEL_DIR / "best_model.pt"
                torch.save(self.model.state_dict(), save_path)
            else:
                self.patience_counter += 1
            
            scheduler.step(val_loss)
            
            if self.patience_counter >= ModelConfig.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Load best model
        best_path = MODEL_DIR / "best_model.pt"
        self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        print(f"\nBest model restored from {best_path}")
        
        return self.history


def main():
    """Main training script"""
    print("\n" + "="*60)
    print("AEGIS - Network Intrusion Detection System")
    print("Training Pipeline")
    print("="*60 + "\n")
    
    # FIX: Use device from config, not hardcoded
    device = torch.device(ModelConfig.device)
    print(f"Device: {device}\n")
    
    # ============================================
    # 1. Data Loading
    # ============================================
    print("STEP 1: Loading Data")
    print("-" * 40)
    
    preprocessor = DataPreprocessor()
    
    try:
        data = preprocessor.load_cicids_data(DATA_DIR)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please ensure CICIDS 2017 CSV files are in: {DATA_DIR}")
        sys.exit(1)
    
    # ============================================
    # 2. Data Preprocessing
    # ============================================
    print("\nSTEP 2: Data Preprocessing")
    print("-" * 40)
    
    X, y, info = preprocessor.preprocess(data, fit=True)
    
    print(f"Dataset Info:")
    print(f"  - Total Samples: {info['n_samples']}")
    print(f"  - Features: {info['n_features']}")
    print(f"  - Class Distribution: {info['class_distribution']}")
    
    # ============================================
    # 3. Train-Validation-Test Split
    # ============================================
    print("\nSTEP 3: Data Splitting")
    print("-" * 40)
    
    from sklearn.model_selection import train_test_split
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TrainingConfig.test_size,
        random_state=TrainingConfig.random_state,
        stratify=y
    )
    
    # Second split: train vs val
    val_ratio = TrainingConfig.validation_size / (1 - TrainingConfig.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio,
        random_state=TrainingConfig.random_state,
        stratify=y_temp
    )
    
    print(f"Training Set: {len(X_train)} samples")
    print(f"Validation Set: {len(X_val)} samples")
    print(f"Test Set: {len(X_test)} samples")
    
    # ============================================
    # 4. Create DataLoaders
    # ============================================
    print("\nSTEP 4: Creating DataLoaders")
    print("-" * 40)
    
    train_dataset = NetworkTrafficDataset(X_train, y_train)
    val_dataset = NetworkTrafficDataset(X_val, y_val)
    test_dataset = NetworkTrafficDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=ModelConfig.batch_size,
        shuffle=True,
        num_workers=TrainingConfig.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=ModelConfig.batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=ModelConfig.batch_size,
        shuffle=False
    )
    
    print(f"DataLoaders created successfully")
    print(f"Batch Size: {ModelConfig.batch_size}")
    
    # ============================================
    # 5. Model Creation
    # ============================================
    print("\nSTEP 5: Creating Model")
    print("-" * 40)
    
    model = create_model(
        model_type="advanced",
        input_size=info["n_features"],
        output_size=2,
        device=device
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: AdvancedNetworkIDS")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # ============================================
    # 6. Training
    # ============================================
    print("\nSTEP 6: Training Model")
    print("-" * 40)
    
    trainer = Trainer(model, device=device)
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=ModelConfig.epochs,
        learning_rate=ModelConfig.learning_rate
    )
    
    # ============================================
    # 7. Model Evaluation
    # ============================================
    print("\nSTEP 7: Model Evaluation")
    print("-" * 40)
    
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_proba = []
    
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == y).sum().item()
            test_total += y.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            proba = torch.nn.functional.softmax(outputs, dim=1)
            all_proba.extend(proba.cpu().numpy())
    
    test_accuracy = 100 * test_correct / test_total
    test_loss = test_loss / len(test_loader)
    
    metrics = calculate_metrics(y_test, np.array(all_preds), np.array(all_proba))
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    if "auc_roc" in metrics:
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")
    
    # ============================================
    # 8. Save Model and Preprocessor
    # ============================================
    print("\nSTEP 8: Saving Model")
    print("-" * 40)
    
    model_path = MODEL_DIR / "aegis_ids_model.pt"
    scaler_path = MODEL_DIR / "scaler.pkl"
    encoder_path = MODEL_DIR / "label_encoder.pkl"
    
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    with open(scaler_path, "wb") as f:
        pickle.dump(preprocessor.scaler, f)
    print(f"Scaler saved to {scaler_path}")
    
    with open(encoder_path, "wb") as f:
        pickle.dump(preprocessor.label_encoder, f)
    print(f"Label Encoder saved to {encoder_path}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()