#!/usr/bin/env python3
"""
Example script demonstrating how to use TabDDPM with the SDE Framework
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from generators.tabddpm_generator import TabDDPMGenerator

def main():
    """Example usage of TabDDPM generator"""
    
    # Load sample data
    data_path = Path("data/raw/diabetes.csv")
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Please ensure you have the diabetes.csv file in data/raw/")
        return
    
    df = pd.read_csv(data_path)
    # Use a smaller subset for testing
    df = df.sample(n=min(5000, len(df)), random_state=42).reset_index(drop=True)
    print(f"Loaded dataset: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Initialize TabDDPM generator
    generator = TabDDPMGenerator(
        output_dir="data/synthetic/tabddpm",
        num_experiments=1
    )
    
    # TabDDPM configuration
    tabddpm_config = {
        "train_size": min(1000, len(df)),  # Use smaller train size
        "eval_model": "catboost",
        "exp_name": "diabetes_experiment",
        "n_sample_seeds": 5,
        "n_eval_seeds": 10,
        # Data preprocessing
        "target_column": "diabetes",  # Target column name
        "categorical_columns": ["gender", "smoking_history"],  # Categorical columns
        # Model parameters
        "hidden_dims": [256, 256],
        "dropout": 0.1,
        "embedding_dim": 128,
        "batch_size": 1024,
        "lr": 0.0002,
        "num_epochs": 100,
        "num_timesteps": 1000,
        "num_samples": len(df)
    }
    
    try:
        # Generate synthetic data
        print("Starting TabDDPM generation...")
        synthetic_data, metrics = generator.fit_and_generate(
            df=df,
            dataset_name="diabetes_custom",
            tabddpm_config=tabddpm_config
        )
        
        print(f"\nGeneration completed!")
        print(f"Synthetic data shape: {synthetic_data.shape}")
        print(f"Execution time: {metrics['execution_time_sec']:.2f} seconds")
        print(f"Peak memory: {metrics['peak_memory_mb']:.2f} MB")
        
        # Display sample of synthetic data
        print("\nSample of synthetic data:")
        print(synthetic_data.head())
        
    except Exception as e:
        print(f"Error during TabDDPM generation: {e}")
        print("Make sure you have:")
        print("1. Run setup_tabddpm.sh to set up the environment")
        print("2. Activated the tabddpm conda environment")
        print("3. Have the TabDDPM repository cloned in the tabddpm/ directory")

if __name__ == "__main__":
    main()
