# Synthetic Tabular Data Evaluation

This repository contains code for a comprehensive evaluation of six synthetic data generation models on tabular datasets. The evaluation covers **data quality**, **privacy**, **machine learning usability**, and **computational complexity**.

---

##  Generative Models Evaluated

- **TVAE** 
- **CTGAN** 
- **CTABGAN**
- **GREAT**
- **RTF**
- **TABDDPM** (Implemented in a separate environment because it depends on an older Python version that is incompatible with the setup used for the other models)

### TabDDPM Integration

TabDDPM (Tabular Diffusion Models) is integrated as a separate module with its own environment configuration. See [TABDDPM_INTEGRATION.md](TABDDPM_INTEGRATION.md) for detailed setup and usage instructions.

**Quick Setup:**
```bash
./setup_tabddpm.sh
conda activate tabddpm
python examples/tabddpm_example.py
```

---

##  Evaluation Criteria

- **Quality**: Statistical similarity to real data
- **Privacy**: Risk of identity disclosure or memorization
- **ML Usability**: How well models trained on synthetic data perform
- **Computational Complexity**: Time and memory usage

---

##  Project Structure

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
│   ├── 04_privacy_evaluation.ipynb (Includes duplicates evaluation)
│   ├── 05_TSTR_evaluation.ipynb
│   └── 06_QPU_eval.ipynb
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

##  Getting Started

### 1. Clone the repository

```bash
https://github.com/akurakova/SDE_clean.git
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

##  Running the Experiments

Each part of the pipeline is modularized:

### Preprocessing:
01_preprocessing.ipynb

### Data Generation + Computational Complexity
02_generate_synthetic.ipynb

### Quality 
03_quality_evaluation

### Privacy
04_privacy_MIA.ipynb

### Usability
05_TSTR_evaluation.py

### QPU Evaluation - Example to illustrate the potential implementation of the framework
06_QPU_eval.ipynb

---

##  Reproducibility Notes

- Random seeds are set where applicable
- All preprocessing is standardized via `src/utils/preprocessing.py`
- Memory profiling is integrated in `02_data_generation.ipynb` using `memory_profiler`

---

##  License

MIT License

---

##  Acknowledgements

This work is part of the research paper:  
**A Comprehensive Evaluation Framework for Synthetic
Medical Tabular Data Generation**  
Submitted to: Journal of Biomedical Informatics, 2025
