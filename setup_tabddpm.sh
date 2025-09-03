#!/bin/bash

# Setup script for TabDDPM integration

echo "Setting up TabDDPM integration..."

# Clone TabDDPM repository
if [ ! -d "tabddpm" ]; then
    echo "Cloning TabDDPM repository..."
    git clone https://github.com/yandex-research/tab-ddpm.git tabddpm
else
    echo "TabDDPM directory already exists, skipping clone..."
fi

# Create conda environment for TabDDPM
echo "Creating TabDDPM conda environment..."
if conda env list | grep -q "tabddpm"; then
    echo "TabDDPM environment already exists, updating..."
    conda env update -f environment_tabddpm.yml
else
    conda env create -f environment_tabddpm.yml
fi

# Activate environment and install TabDDPM requirements
echo "Installing TabDDPM dependencies..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate tabddpm
cd tabddpm

# Install requirements with updated versions
echo "Installing TabDDPM requirements..."
pip install -r requirements.txt || {
    echo "Some requirements failed to install, trying with updated versions..."
    echo "Installing core dependencies with compatible versions..."
    pip install catboost>=1.2.0
    pip install category-encoders>=2.3.0
    pip install dython>=0.5.1
    pip install icecream>=2.1.2
    pip install libzero>=0.0.8
    pip install numpy>=1.21.4
    pip install optuna>=2.10.1
    pip install pandas>=1.3.4
    pip install pyarrow>=6.0.0
    pip install rtdl>=0.0.9
    pip install scikit-learn>=1.0.2
    pip install scipy>=1.7.2
    pip install skorch>=0.11.0
    pip install tomli-w>=0.4.0
    pip install tomli>=1.2.2
    pip install tqdm>=4.62.3
    pip install imbalanced-learn>=0.7.0
    pip install rdt>=0.6.4
}

# Set up environment variables
echo "Setting up environment variables..."
export REPO_DIR=$(pwd)
export PROJECT_DIR=$(pwd)
conda env config vars set PYTHONPATH=${PYTHONPATH}:${REPO_DIR} || true
conda env config vars set PROJECT_DIR=${REPO_DIR} || true

# Create data directory structure
echo "Creating data directory structure..."
mkdir -p data/synthetic/tabddpm

echo "TabDDPM setup complete!"
echo ""
echo "To use TabDDPM:"
echo "1. Activate the environment: conda activate tabddpm"
echo "2. Run your synthetic data generation scripts"
echo "3. Generated data will be saved to data/synthetic/tabddpm/"
