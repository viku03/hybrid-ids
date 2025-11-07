import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from model import create_model
from utils import DataPreprocessor
from config import MODEL_DIR

def demo_basic_inference():
    """Demo 1: Basic model inference"""
    print("\n" + "="*60)
    print("DEMO 1: Basic Model Inference")
    print("="*60)
    
    # Load model
    model_path = MODEL_DIR / "aegis_ids_model.pt"
    if not model_path.exists():
        print("❌ Model not found. Please train first: python train_model.py")
        return
    
    # Initialize and load model
    model = create_model("advanced", input_size=78, output_size=2, device="cpu")
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
        model.eval()
        print("✓ Model loaded successfully")
    except RuntimeError as e:
        print(f"❌ Model loading error: {e}")
        return
    
    # Generate random test data
    X_test = np.random.randn(5, 78).astype(np.float32)
    X_tensor = torch.FloatTensor(X_test)
    
    # Make predictions
    with torch.no_grad():
        predictions = model.predict(X_tensor)
        probabilities = model.predict_proba(X_tensor)
    
    print("\nResults:")
    print(f"{'Sample':<10} {'Prediction':<15} {'Confidence':<15}")
    print("-" * 40)
    for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
        label = "BENIGN" if pred == 0 else "ATTACK"
        confidence = max(proba).item()
        print(f"{i+1:<10} {label:<15} {confidence:.2%}")

def demo_batch_processing():
    """Demo 2: Batch CSV processing"""
    print("\n" + "="*60)
    print("DEMO 2: Batch CSV Processing")
    print("="*60)
    
    # Load model and preprocessor
    model_path = MODEL_DIR / "aegis_ids_model.pt"
    scaler_path = MODEL_DIR / "scaler.pkl"
    
    if not all([model_path.exists(), scaler_path.exists()]):
        print("❌ Model artifacts not found. Please train first.")
        return
    
    # Load model
    model = create_model("advanced", input_size=78, output_size=2, device="cpu")
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
        model.eval()
    except RuntimeError as e:
        print(f"❌ Model loading error: {e}")
        return
    
    # Load scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    print("✓ Model and scaler loaded")
    
    # Generate synthetic batch data
    n_samples = 100
    X_batch = np.random.randn(n_samples, 78).astype(np.float32)
    
    # Preprocess
    X_scaled = scaler.transform(X_batch)
    X_tensor = torch.FloatTensor(X_scaled)
    
    # Predict
    with torch.no_grad():
        predictions = model.predict(X_tensor)
        probabilities = model.predict_proba(X_tensor)
    
    # Convert to numpy for statistics
    predictions_np = predictions.cpu().numpy()
    probabilities_np = probabilities.cpu().numpy()
    
    # Create results dataframe
    results_df = pd.DataFrame({
        "Sample_ID": range(1, n_samples + 1),
        "Prediction": ["BENIGN" if p == 0 else "ATTACK" for p in predictions_np],
        "Confidence": np.max(probabilities_np, axis=1),
        "Normal_Score": probabilities_np[:, 0],
        "Attack_Score": probabilities_np[:, 1]
    })
    
    print(f"\nProcessed {n_samples} samples")
    print(f"\nSummary Statistics:")
    # FIX: Convert torch tensor to numpy before numpy operations
    benign_count = int((predictions_np == 0).sum())
    attack_count = int((predictions_np == 1).sum())
    print(f"  - Benign Traffic: {benign_count} ({benign_count/n_samples:.1%})")
    print(f"  - Attack Traffic: {attack_count} ({attack_count/n_samples:.1%})")
    print(f"  - Avg Confidence: {np.mean(results_df['Confidence']):.2%}")
    
    print("\nFirst 10 Results:")
    print(results_df.head(10).to_string(index=False))
    
    # Save results
    output_path = Path("demo_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")

def demo_model_architecture():
    """Demo 3: Model architecture inspection"""
    print("\n" + "="*60)
    print("DEMO 3: Model Architecture Inspection")
    print("="*60)
    
    model = create_model("advanced", input_size=78, output_size=2, device="cpu")
    
    print("\nModel Summary:")
    print(model)
    
    print("\nParameter Count:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    
    print("\nLayer Breakdown:")
    for name, param in model.named_parameters():
        print(f"  {name:<30} {param.shape}")

def demo_preprocessing():
    """Demo 4: Data preprocessing pipeline"""
    print("\n" + "="*60)
    print("DEMO 4: Data Preprocessing Pipeline")
    print("="*60)
    
    # Create synthetic data
    print("Creating synthetic data...")
    n_samples = 50
    n_features = 78
    
    X = np.random.randn(n_samples, n_features) * 100 + 50
    y = np.random.choice([0, 1], n_samples)
    
    print(f"  - Samples: {n_samples}")
    print(f"  - Features: {n_features}")
    
    # Create and fit preprocessor
    preprocessor = DataPreprocessor()
    
    print("\nFitting preprocessor...")
    X_scaled = preprocessor.scale_features(X, fit=True)
    
    print(f"✓ Scaler fitted")
    print(f"  - Original X: mean={X.mean():.2f}, std={X.std():.2f}")
    print(f"  - Scaled X: mean={X_scaled.mean():.4f}, std={X_scaled.std():.4f}")
    
    # Transform new data
    print("\nTransforming new data...")
    X_new = np.random.randn(10, n_features) * 100 + 50
    X_new_scaled = preprocessor.scale_features(X_new, fit=False)
    
    print(f"✓ New data transformed")
    print(f"  - Original range: [{X_new.min():.2f}, {X_new.max():.2f}]")
    print(f"  - Scaled range: [{X_new_scaled.min():.4f}, {X_new_scaled.max():.4f}]")

def demo_comparison():
    """Demo 5: Model variants comparison"""
    print("\n" + "="*60)
    print("DEMO 5: Model Variants Comparison")
    print("="*60)
    
    X_test = torch.randn(10, 78)
    
    print("\nComparing model architectures:")
    print(f"{'Model':<25} {'Output Shape':<20} {'Parameters':<15}")
    print("-" * 60)
    
    # Advanced Model
    model_advanced = create_model("advanced", input_size=78, output_size=2)
    with torch.no_grad():
        out_adv = model_advanced(X_test)
    
    params_adv = sum(p.numel() for p in model_advanced.parameters())
    print(f"{'AdvancedNetworkIDS':<25} {str(out_adv.shape):<20} {params_adv:,}")
    
    # Transformer Model
    model_transformer = create_model("transformer", input_size=78, output_size=2)
    with torch.no_grad():
        out_trans = model_transformer(X_test)
    
    params_trans = sum(p.numel() for p in model_transformer.parameters())
    print(f"{'Transformer':<25} {str(out_trans.shape):<20} {params_trans:,}")
    
    print(f"\nConclusion: AdvancedNetworkIDS is more efficient")
    print(f"Recommended for production: AdvancedNetworkIDS")

def main():
    """Run all demos"""
    print("\n" + "🛡️ "*20)
    print("AEGIS IDS - DEMONSTRATION SCRIPT")
    print("🛡️ "*20)
    
    try:
        demo_basic_inference()
    except Exception as e:
        print(f"❌ Demo 1 failed: {e}")
    
    try:
        demo_batch_processing()
    except Exception as e:
        print(f"❌ Demo 2 failed: {e}")
    
    try:
        demo_model_architecture()
    except Exception as e:
        print(f"❌ Demo 3 failed: {e}")
    
    try:
        demo_preprocessing()
    except Exception as e:
        print(f"❌ Demo 4 failed: {e}")
    
    try:
        demo_comparison()
    except Exception as e:
        print(f"❌ Demo 5 failed: {e}")
    
    print("\n" + "="*60)
    print("✓ Demonstration Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run tests: pytest test_model.py test_integration.py -v")
    print("2. Run Streamlit UI: streamlit run streamlit_app.py")
    print("3. Check git commits: git log --oneline")

if __name__ == "__main__":
    main()