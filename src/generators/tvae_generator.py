import time
import tracemalloc
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sdv.single_table import TVAESynthesizer
from sdv.metadata import Metadata

# Optional GPU tracking (safe if torch is missing or CPU-only)
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _match_format(syn_df: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
    """Align columns and best-effort dtypes to match the real dataframe."""
    # Keep only real columns
    syn_df = syn_df[[c for c in real_df.columns if c in syn_df.columns]].copy()
    # Add any missing columns as NaN
    for c in real_df.columns:
        if c not in syn_df.columns:
            syn_df[c] = np.nan
    # Reorder to real_df
    syn_df = syn_df[real_df.columns]
    # Best-effort dtype alignment
    for c in real_df.columns:
        try:
            syn_df[c] = syn_df[c].astype(real_df[c].dtype, errors="ignore")
        except Exception:
            pass
    return syn_df


class TVAESynthesizerWrapper:
    def __init__(self, output_dir: str = "../data/synthetic/tvae"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _gpu_reset(self):
        if _HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def _gpu_peak_mb(self) -> float:
        if _HAS_TORCH and torch.cuda.is_available():
            try:
                return torch.cuda.max_memory_allocated() / (1024 ** 2)
            except Exception:
                return float("nan")
        return float("nan")

    def fit_and_generate(self, df: pd.DataFrame, dataset_name: str, **model_params) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        print("\nGenerating with TVAE")
        print("Detecting metadata from input dataframe...")
        metadata = Metadata.detect_from_dataframe(data=df)

        print("Initializing TVAE synthesizer...")
        synthesizer = TVAESynthesizer(metadata, **model_params)

        # ---------------------------
        # Train: time + CPU + GPU
        # ---------------------------
        print("Starting model training...")
        tracemalloc.start()
        self._gpu_reset()
        t0 = time.time()

        synthesizer.fit(df)

        train_time = time.time() - t0
        current_bytes, peak_bytes_train = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_cpu_mb_train = peak_bytes_train / (1024 ** 2)
        peak_gpu_mb_train = self._gpu_peak_mb()
        print("Training complete.")

        # ---------------------------
        # Sample: time + CPU + GPU
        # ---------------------------
        print("Generating synthetic data...")
        tracemalloc.start()
        self._gpu_reset()
        s0 = time.time()

        synthetic_data = synthesizer.sample(num_rows=len(df))

        sample_time = time.time() - s0
        current_bytes_s, peak_bytes_sample = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_cpu_mb_sample = peak_bytes_sample / (1024 ** 2)
        peak_gpu_mb_sample = self._gpu_peak_mb()

        # Align schema/dtypes to the real dataframe
        synthetic_data = _match_format(synthetic_data, df)

        # ---------------------------
        # Save + report
        # ---------------------------
        out_path = self.output_dir / f"{dataset_name}_tvae.csv"
        synthetic_data.to_csv(out_path, index=False)

        print(f"Synthetic data saved to: {out_path}")
        print(f"Training time: {train_time:.2f} seconds")
        print(f"Peak CPU memory during training: {peak_cpu_mb_train:.2f} MB")
        if not np.isnan(peak_gpu_mb_train):
            print(f"Peak GPU VRAM during training: {peak_gpu_mb_train:.2f} MB")

        print(f"Sampling time: {sample_time:.2f} seconds")
        print(f"Peak CPU memory during sampling: {peak_cpu_mb_sample:.2f} MB")
        if not np.isnan(peak_gpu_mb_sample):
            print(f"Peak GPU VRAM during sampling: {peak_gpu_mb_sample:.2f} MB")

        metrics = {
            "execution_time_sec": train_time,
            #"sampling_time_sec": sample_time,
            "peak_memory_mb": peak_cpu_mb_train,
            #"peak_cpu_mb_sample": peak_cpu_mb_sample,
            #"peak_gpu_mb_train": peak_gpu_mb_train,     # may be NaN if no CUDA
            #"peak_gpu_mb_sample": peak_gpu_mb_sample,   # may be NaN if no CUDA
        }

        return synthetic_data, metrics
