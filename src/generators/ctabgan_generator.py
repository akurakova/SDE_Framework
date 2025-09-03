import os
import time
import tracemalloc
import io
import contextlib
import numpy as np
import pandas as pd
from pathlib import Path
from model.ctabgan import CTABGAN
from src.utils.postprocess import match_format

# Optional GPU tracking (safe if torch unavailable)
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


class CTABGANSynthesizerWrapper:
    def __init__(self, output_dir: str = "data/synthetic/ctabgan", num_experiments: int = 1):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_experiments = num_experiments

    def fit_and_generate(self, df: pd.DataFrame, dataset_name: str, ctabgan_config: dict):
        print("Initializing CTABGAN synthesizer...")
        synthesizer = CTABGAN(
            raw_csv_path=ctabgan_config["raw_csv_path"],
            categorical_columns=ctabgan_config.get("categorical_columns", []),
            log_columns=ctabgan_config.get("log_columns", []),
            mixed_columns=ctabgan_config.get("mixed_columns", {}),
            general_columns=ctabgan_config.get("general_columns", []),
            non_categorical_columns=ctabgan_config.get("non_categorical_columns", []),
            integer_columns=ctabgan_config.get("integer_columns", []),
            problem_type=ctabgan_config.get("problem_type", {})
        )

        logs = []
        last = {}

        for i in range(self.num_experiments):
            print(f"Running experiment {i+1}/{self.num_experiments}...")

            # ---------------------------
            # TRAIN: time + CPU + (opt) GPU
            # ---------------------------
            tracemalloc.start()
            _gpu_reset()
            t0 = time.time()

            # CTABGAN can be verbose; silence if needed:
            with contextlib.redirect_stdout(io.StringIO()):
                synthesizer.fit()

            train_time = time.time() - t0
            _, peak_bytes_train = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_cpu_mb_train = peak_bytes_train / (1024 * 1024)
            peak_gpu_mb_train = _gpu_peak_mb()

            # ---------------------------
            # SAMPLE: time + CPU + (opt) GPU
            # ---------------------------
            tracemalloc.start()
            _gpu_reset()
            s0 = time.time()

            synthetic_data = synthesizer.generate_samples()

            sample_time = time.time() - s0
            _, peak_bytes_sample = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_cpu_mb_sample = peak_bytes_sample / (1024 * 1024)
            peak_gpu_mb_sample = _gpu_peak_mb()

            # Align to real schema/dtypes and save
            synthetic_data = match_format(synthetic_data, df)
            file_path = self.output_dir / f"{dataset_name}_ctabgan_{i}.csv"
            synthetic_data.to_csv(file_path, index=False)

            print(f"Saved: {file_path}")
            print(f"Training time: {train_time:.2f} s | Peak CPU(train): {peak_cpu_mb_train:.2f} MB")
            if not np.isnan(peak_gpu_mb_train):
                print(f"Peak GPU(train): {peak_gpu_mb_train:.2f} MB")
            print(f"Sampling time: {sample_time:.2f} s | Peak CPU(sample): {peak_cpu_mb_sample:.2f} MB")
            if not np.isnan(peak_gpu_mb_sample):
                print(f"Peak GPU(sample): {peak_gpu_mb_sample:.2f} MB")

            logs.append({
                "experiment": i,
                "execution_time_sec": train_time,
                "sampling_time_sec": sample_time,
                "peak_cpu_mb_train": peak_cpu_mb_train,
                "peak_cpu_mb_sample": peak_cpu_mb_sample,
                "peak_gpu_mb_train": peak_gpu_mb_train,
                "peak_gpu_mb_sample": peak_gpu_mb_sample,
                "n_samples": len(synthetic_data),
            })

            last = {
                "synthetic": synthetic_data.copy(),
                "train_time": train_time,
                "peak_cpu_train": peak_cpu_mb_train,
                "peak_gpu_train": peak_gpu_mb_train,
                "sample_time": sample_time,
                "peak_cpu_sample": peak_cpu_mb_sample,
                "peak_gpu_sample": peak_gpu_mb_sample,
            }

        # Return last run’s artifacts + metrics to mirror your other wrappers
        return last["synthetic"], {
            "execution_time_sec": last["train_time"],
            "peak_memory_mb": last["peak_cpu_train"],
            "sampling_time_sec": last["sample_time"],
            "peak_cpu_mb_sample": last["peak_cpu_sample"],
            "peak_gpu_mb_train": last["peak_gpu_train"],
            "peak_gpu_mb_sample": last["peak_gpu_sample"],
            "runs": logs,
        }
