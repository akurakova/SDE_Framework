import pandas as pd
import time
import tracemalloc
from pathlib import Path
from realtabformer import REaLTabFormer
import io


class RTFGeneratorWrapper:
    def __init__(self, output_dir: str = "data/synthetic/rtf"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fit_and_generate(self, df: pd.DataFrame, dataset_name: str):
        print("Initializing REaLTabFormer model...")

        model = REaLTabFormer(
            model_type="tabular",
            batch_size=64,
            epochs=30,
            gradient_accumulation_steps=4,
            mask_rate=0,
            logging_steps=100,
        )

        print("Starting training...")
        tracemalloc.start()
        start_time = time.time()

        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(df, num_bootstrap=10)

        print("Generating synthetic data...")
        synthetic_data = model.sample(n_samples=len(df))

        end_time = time.time()
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        output_file = self.output_dir / f"{dataset_name}_rtf.csv"
        synthetic_data.to_csv(output_file, index=False)

        print(f"Saved synthetic data to: {output_file}")
        print(f"Execution Time: {end_time - start_time:.2f} seconds")
        print(f"Peak Memory Usage: {peak_memory / (1024 * 1024):.2f} MB")

        return synthetic_data.copy(), {
            "execution_time_sec": end_time - start_time,
            "peak_memory_mb": peak_memory / (1024 * 1024),
        }
