import os
import time
import tracemalloc
import pandas as pd
from pathlib import Path
from model.ctabgan import CTABGAN

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

        log = []
        for i in range(self.num_experiments):
            print(f"Running experiment {i+1}/{self.num_experiments}...")

            start_time = time.time()
            tracemalloc.start()

            synthesizer.fit()

            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            end_time = time.time()

            synthetic_data = synthesizer.generate_samples()
            file_path = self.output_dir / f"{dataset_name}_ctabgan_{i}.csv"
            synthetic_data.to_csv(file_path, index=False)

            print(f"Saved: {file_path}")
            print(f"Training time: {end_time - start_time:.2f} seconds")
            print(f"Peak memory: {peak_mem / (1024 * 1024):.2f} MB")

            log.append({
                "experiment": i,
                "execution_time_sec": end_time - start_time,
                "peak_memory_mb": peak_mem / (1024 * 1024),
                "n_samples": len(synthetic_data)
            })

        return synthetic_data.copy(), {
        "execution_time_sec": end_time - start_time,
        "peak_memory_mb": peak_mem / (1024 * 1024),
    }