# Synthetic Tabular Data Evaluation

This repository contains code for a comprehensive evaluation of six synthetic data generation models on tabular datasets. The evaluation covers **data quality**, **privacy**, **machine learning usability**, and **computational complexity**.

---

## 📌 Generative Models Evaluated

- **TVAE** (Tabular Variational Autoencoder)
- **CTGAN** (Conditional Tabular GAN)
- **CTABGAN**
- **GREAT**
- **RTF**
- **TABDDPM**

---

## 📊 Evaluation Criteria

- **Quality**: Statistical similarity to real data
- **Privacy**: Risk of identity disclosure or memorization
- **ML Usability**: How well models trained on synthetic data perform
- **Computational Complexity**: Time and memory usage

---

## 📁 Project Structure

```
synthetic-eval/
│
├── data/                    # Real and synthetic datasets
│   ├── raw/                 # Original datasets
│   └── synthetic/           # Synthetic datasets
│   └── processed/           # Preprocessed data for models
│
├── notebooks/               # Jupyter notebooks for analysis
│   ├── 01_preprocessing.ipynb
│   ├── 02_data_generation.ipynb
│   ├── 03_evaluation_quality.ipynb
│   ├── 04_evaluation_privacy_MIA.ipynb (Includes duplicates evaluation)
│   └── 05_TSTR_evaluation.ipynb
│
├── src/
│   ├── generators/          # Scripts to train each generative model
│   ├── evaluation/          # Evaluation metric implementations
│   └── utils/               # Preprocessing and helper functions
│
├── results/                 # Logs, figures, and output metrics
│   ├── figures/
│   └── logs/
│
├── environment.yml          # Conda environment for reproducibility
├── LICENSE
├── README.md
└── .gitignore
```

---

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/synthetic-eval-clean.git
cd synthetic-eval-clean
```

### 2. Create and activate the environment

Using conda:

```bash
conda env create -f environment.yml
conda activate synthetic-eval
```

Or using pip:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Experiments

Each part of the pipeline is modularized:

### Data Generation

```bash
python src/generation/run_ctgan.py
python src/generation/run_tvae.py
# ... other models
```

### Evaluation

```bash
python src/evaluation/quality.py
python src/evaluation/privacy.py
python src/evaluation/usability.py
```

Notebooks (`notebooks/*.ipynb`) provide visualization and result comparison.

---

## 🧪 Reproducibility Notes

- Random seeds are set where applicable
- All preprocessing is standardized via `src/utils/preprocessing.py`
- Memory profiling is integrated in `complexity.py` using `memory_profiler`

---

## 📜 License

MIT License

---

## 🙋‍♀️ Acknowledgements

This work is part of the research paper:  
**A Comprehensive Evaluation Framework for Synthetic
Medical Tabular Data Generation**  
Submitted to: Journal of Biomedical Informatics, 2025
