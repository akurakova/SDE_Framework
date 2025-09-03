# TabDDPM Integration

This document describes how to integrate and use TabDDPM (Tabular Diffusion Models) with the SDE Framework.

## Overview

TabDDPM is a diffusion model approach for generating synthetic tabular data. This integration allows you to use TabDDPM alongside other synthetic data generators in the framework while maintaining consistent output formats and evaluation procedures.

## Setup

### 1. Automatic Setup (Recommended)

Run the setup script to automatically configure TabDDPM:

```bash
chmod +x setup_tabddpm.sh
./setup_tabddpm.sh
```

This script will:
- Clone the TabDDPM repository from GitHub
- Create a separate conda environment (`tabddpm`)
- Install all required dependencies
- Set up environment variables
- Create the necessary directory structure

### 2. Manual Setup

If you prefer manual setup:

1. **Clone TabDDPM repository:**
   ```bash
   git clone https://github.com/yandex-research/tab-ddpm.git tabddpm
   ```

2. **Create conda environment:**
   ```bash
   conda env create -f environment_tabddpm.yml
   ```

3. **Activate environment and install dependencies:**
   ```bash
   conda activate tabddpm
   cd tabddpm
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   export REPO_DIR=$(pwd)
   export PROJECT_DIR=$(pwd)
   conda env config vars set PYTHONPATH=${PYTHONPATH}:${REPO_DIR}
   conda env config vars set PROJECT_DIR=${REPO_DIR}
   ```

## Usage

### Basic Usage

```python
import pandas as pd
from src.generators.tabddpm_generator import TabDDPMGenerator

# Load your data
df = pd.read_csv("data/raw/your_dataset.csv")

# Initialize TabDDPM generator
generator = TabDDPMGenerator(
    output_dir="data/synthetic/tabddpm",
    num_experiments=1
)

# Configure TabDDPM
config = {
    "train_size": len(df),
    "eval_model": "catboost",  # or "mlp"
    "exp_name": "my_experiment",
    "hidden_dims": [256, 256],
    "dropout": 0.1,
    "batch_size": 1024,
    "lr": 0.0002,
    "num_epochs": 100,
    "num_timesteps": 1000,
    "num_samples": len(df),
    "categorical_columns": [],  # Add categorical column names
    "numerical_columns": list(df.columns),
    "mixed_columns": {}
}

# Generate synthetic data
synthetic_data, metrics = generator.fit_and_generate(
    df=df,
    dataset_name="your_dataset",
    tabddpm_config=config
)
```

### Configuration Options

The TabDDPM configuration supports the following parameters:

#### Model Parameters
- `hidden_dims`: List of hidden layer dimensions (default: [256, 256])
- `dropout`: Dropout rate (default: 0.1)
- `embedding_dim`: Embedding dimension (default: 128)

#### Training Parameters
- `batch_size`: Training batch size (default: 1024)
- `lr`: Learning rate (default: 0.0002)
- `num_epochs`: Number of training epochs (default: 100)
- `num_timesteps`: Number of diffusion timesteps (default: 1000)

#### Data Parameters
- `categorical_columns`: List of categorical column names
- `numerical_columns`: List of numerical column names
- `mixed_columns`: Dictionary of mixed column configurations

#### Sampling Parameters
- `num_samples`: Number of synthetic samples to generate
- `train_size`: Number of training samples to use

## Output Structure

Generated synthetic data is saved in the following structure:

```
data/synthetic/tabddpm/
├── dataset_name_tabddpm_0.csv
├── dataset_name_tabddpm_1.csv
└── ...
```

Each file contains synthetic data generated from a single experiment run.

## Integration with Framework

The TabDDPM generator follows the same interface as other generators in the framework:

1. **Consistent Output Format**: Synthetic data is saved in the same CSV format as other generators
2. **Post-processing**: Uses the same `match_format` function to ensure consistency
3. **Metrics Tracking**: Provides execution time and memory usage metrics
4. **Directory Structure**: Follows the established `data/synthetic/` pattern

## Example Script

See `examples/tabddpm_example.py` for a complete working example using the diabetes dataset.

## Troubleshooting

### Common Issues

1. **TabDDPM directory not found**
   - Ensure you've run the setup script or manually cloned the repository
   - Check that the `tabddpm/` directory exists in your project root

2. **Environment not activated**
   - Make sure to activate the TabDDPM environment: `conda activate tabddpm`
   - Verify the environment is active: `conda info --envs`

3. **Missing dependencies**
   - Reinstall requirements: `pip install -r requirements.txt`
   - Check PyTorch installation: `python -c "import torch; print(torch.__version__)"`

4. **CUDA issues**
   - Ensure you have compatible CUDA drivers installed
   - Consider using CPU-only PyTorch if GPU is not available

### Performance Tips

1. **GPU Usage**: TabDDPM benefits significantly from GPU acceleration
2. **Batch Size**: Adjust batch size based on available memory
3. **Number of Epochs**: Start with fewer epochs for testing, increase for better quality
4. **Timesteps**: More timesteps generally improve quality but increase training time

## References

- [TabDDPM Paper](https://arxiv.org/abs/2209.15421)
- [TabDDPM Repository](https://github.com/yandex-research/tab-ddpm)
- [Original Implementation](https://github.com/yandex-research/tab-ddpm.git)
