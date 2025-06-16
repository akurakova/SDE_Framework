import pandas as pd
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

def preprocess_and_split(df: pd.DataFrame, dataset_name: str, project_root: str = "..", test_size=0.2, random_state=42):
    # Drop ID column if it exists (case-insensitive)
    id_columns = [col for col in df.columns if col.lower() == "id"]
    if id_columns:
        df = df.drop(columns=id_columns)

    # Drop rows with missing values
    df = df.dropna()

    # Train/test split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

     # Paths relative to project root
    processed_path = Path(project_root) / "data" / "processed"
    processed_path.mkdir(parents=True, exist_ok=True)
    train_path = processed_path / f"{dataset_name}_train.csv"
    test_path = processed_path / f"{dataset_name}_test.csv"

    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved to:\n - {train_path}\n - {test_path}")
    return train_path, test_path
