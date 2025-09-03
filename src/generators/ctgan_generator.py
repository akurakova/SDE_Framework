import pandas as pd
import time
from pathlib import Path
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata


try:
    from src.utils.postprocess import match_format as _match_format
except Exception:
    _match_format = None

# CPU memory profiling

from memory_profiler import memory_usage


class CTGANSynthesizerWrapper:
    def __init__(self, output_dir: str = "data/synthetic/ctgan"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _measure_cpu_peak(self, fn, interval: float = 0.05):
        """
        Run callable `fn()` while measuring peak CPU RAM (MB) and wall time (s).
        Returns: (peak_cpu_mb: float, retval: Any, wall_time_s: float)
        """
        t0 = time.perf_counter()
        peak_mb, retval = memory_usage(
            (fn, (), {}),
            max_usage=True,
            retval=True,
            include_children=True,
            multiprocess=True,
            backend='psutil',
            interval=interval,
        )
        t1 = time.perf_counter()
        return float(peak_mb), retval, float(t1 - t0)

    def fit_and_generate(self, df: pd.DataFrame, dataset_name: str, **model_params):
        # ---- Metadata & model init ----
        print("Detecting metadata from input dataframe...")
        metadata = Metadata.detect_from_dataframe(df)

        print("Initializing CTGAN synthesizer...")
        synthesizer = CTGANSynthesizer(metadata, **model_params)

        # ---- Optional GPU instrumentation (PyTorch) ----
        has_cuda = False
        gpu_peak_train_mb = None
        gpu_peak_sample_mb = None
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except Exception:
            has_cuda = False

        # ---- TRAIN: time + CPU peak + optional GPU peak ----
        print("Starting model training...")
        if has_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        cpu_peak_train_mb, _, train_time_s = self._measure_cpu_peak(lambda: synthesizer.fit(df))

        if has_cuda:
            torch.cuda.synchronize()
            gpu_peak_train_mb = torch.cuda.max_memory_allocated() / (1024**2)

        print("Training complete.")

        # ---- SAMPLE: time + CPU peak + optional GPU peak ----
        print("Generating synthetic data...")
        if has_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        cpu_peak_sample_mb, synthetic_data, sample_time_s = self._measure_cpu_peak(
            lambda: synthesizer.sample(num_rows=len(df))
        )

        if has_cuda:
            torch.cuda.synchronize()
            gpu_peak_sample_mb = torch.cuda.max_memory_allocated() / (1024**2)


        if _match_format is not None:
            try:
                synthetic_data = _match_format(synthetic_data, df)
            except Exception as e:
                print(f"match_format failed, returning raw synthetic data. Reason: {e}")

        # ---- Save to file ----
        out_path = self.output_dir / f"{dataset_name}_ctgan.csv"
        synthetic_data.to_csv(out_path, index=False)

        # ---- Printouts
        print(f"Synthetic data saved to: {out_path}")
        print(f"Training time: {train_time_s:.2f} seconds")
        print(f"Peak CPU memory during training: {cpu_peak_train_mb:.2f} MB")
        if gpu_peak_train_mb is not None:
            print(f"Peak GPU VRAM during training: {gpu_peak_train_mb:.2f} MB")
        print(f"Sampling time: {sample_time_s:.2f} seconds")
        print(f"Peak CPU memory during sampling: {cpu_peak_sample_mb:.2f} MB")
        if gpu_peak_sample_mb is not None:
            print(f"Peak GPU VRAM during sampling: {gpu_peak_sample_mb:.2f} MB")


        metrics = {
            # preserved keys for compatibility
            "execution_time_sec": train_time_s,
            "peak_memory_mb": cpu_peak_train_mb,

            # additional detail
            "sample_time_sec": sample_time_s,
            "peak_cpu_train_mb": cpu_peak_train_mb,
            "peak_cpu_sample_mb": cpu_peak_sample_mb,
            "peak_gpu_train_mb": gpu_peak_train_mb,
            "peak_gpu_sample_mb": gpu_peak_sample_mb,
            "output_path": str(out_path),
        }

        return synthetic_data, metrics
