import time
import tracemalloc
from pathlib import Path
import io
import contextlib
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from be_great import GReaT

try:
    from src.utils.postprocess import match_format as _external_match_format
except Exception:
    _external_match_format = None


def _match_format_local(syn_df: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
    syn_df = syn_df[[c for c in real_df.columns if c in syn_df.columns]].copy()
    for c in real_df.columns:
        if c not in syn_df.columns:
            syn_df[c] = np.nan
    syn_df = syn_df[real_df.columns]
    for c in real_df.columns:
        try:
            syn_df[c] = syn_df[c].astype(real_df[c].dtype, errors="ignore")
        except Exception:
            pass
    return syn_df


def _match_format(syn_df: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
    if _external_match_format is not None:
        try:
            return _external_match_format(syn_df, real_df)
        except Exception:
            pass
    return _match_format_local(syn_df, real_df)


# Optional GPU tracking
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _gpu_reset():
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def _gpu_peak_mb() -> float:
    if _HAS_TORCH and torch.cuda.is_available():
        try:
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
        except Exception:
            return float("nan")
    return float("nan")


class GREATSynthesizerWrapper:
    def __init__(self, output_dir: str = "data/synthetic/great"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fit_and_generate(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        **model_params
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        print("Initializing GReaT synthesizer...")
        default_epochs = 5 if str(dataset_name).lower() == "diabetes" else 100
        # Defaults you had; allow notebook to override via **model_params
        default_params = dict(
            llm="distilgpt2",
            batch_size=64,
            epochs=default_epochs,
            save_steps=10000,
        )
        used_params = {**default_params, **model_params}

        # Guided sampling toggled via param (default True)
        guided_sampling = used_params.pop("guided_sampling", True)

        # Instantiate model
        model = GReaT(**used_params)

        # ---------------------------
        # TRAIN: time + CPU + (optional) GPU
        # ---------------------------
        print("Starting training...")
        tracemalloc.start()
        _gpu_reset()
        t0 = time.time()

        # GReaT can be verbose; silence stdout if desired
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(df)

        train_time = time.time() - t0
        _, peak_bytes_train = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_cpu_mb_train = peak_bytes_train / (1024 ** 2)
        peak_gpu_mb_train = _gpu_peak_mb()
        print("Training complete.")

        # ---------------------------
        # SAMPLE: time + CPU + (optional) GPU
        # ---------------------------
        print("Generating synthetic data...")
        tracemalloc.start()
        _gpu_reset()
        s0 = time.time()

        synthetic_data = model.sample(n_samples=len(df), guided_sampling=guided_sampling)

        sample_time = time.time() - s0
        _, peak_bytes_sample = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_cpu_mb_sample = peak_bytes_sample / (1024 ** 2)
        peak_gpu_mb_sample = _gpu_peak_mb()

        # Align schema / dtypes
        synthetic_data = _match_format(synthetic_data, df)

        # ---------------------------
        # SAVE + REPORT
        # ---------------------------
        out_path = self.output_dir / f"{dataset_name}_great.csv"
        synthetic_data.to_csv(out_path, index=False)

        print(f"Saved synthetic data to: {out_path}")
        print(f"Training time: {train_time:.2f} seconds")
        print(f"Peak CPU memory during training: {peak_cpu_mb_train:.2f} MB")
        if not np.isnan(peak_gpu_mb_train):
            print(f"Peak GPU VRAM during training: {peak_gpu_mb_train:.2f} MB")
        print(f"Sampling time: {sample_time:.2f} seconds")
        print(f"Peak CPU memory during sampling: {peak_cpu_mb_sample:.2f} MB")
        if not np.isnan(peak_gpu_mb_sample):
            print(f"Peak GPU VRAM during sampling: {peak_gpu_mb_sample:.2f} MB")

        metrics = {
            "execution_time_sec": train_time,          # training time
            "peak_memory_mb": peak_cpu_mb_train,       # peak CPU RAM during training
            "sampling_time_sec": sample_time,
            "peak_cpu_mb_sample": peak_cpu_mb_sample,
            "peak_gpu_mb_train": peak_gpu_mb_train,
            "peak_gpu_mb_sample": peak_gpu_mb_sample,
            "output_path": str(out_path),
            "used_params": used_params,
            "guided_sampling": guided_sampling,
        }

        return synthetic_data.copy(), metrics
