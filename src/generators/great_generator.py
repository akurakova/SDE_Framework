
import pandas as pd
import time
import tracemalloc
from pathlib import Path
from be_great import GReaT
from src.utils.postprocess import match_format

class GREATSynthesizerWrapper:
    def __init__(self, output_dir: str = "data/synthetic/great"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fit_and_generate(self, df: pd.DataFrame, dataset_name: str):
        print("Initializing GReaT synthesizer...")

        model = GReaT(
            llm='distilgpt2',
            batch_size=64,
            epochs=100,
            save_steps=10000
        )

        print("Starting training...")
        tracemalloc.start()
        start_time = time.time()

        model.fit(df)

        end_time = time.time()
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

      #  try:
        print("Attempting guided sampling...")
        synthetic_data = model.sample(n_samples=len(df), guided_sampling=True)
        synthetic_data = match_format(synthetic_data, df)

        output_file = self.output_dir / f"{dataset_name}_great.csv"
        synthetic_data.to_csv(output_file, index=False)

        print(f"Saved synthetic data to: {output_file}")
        print(f"Execution Time: {end_time - start_time:.2f} seconds")
        print(f"Peak Memory Usage: {peak_memory / (1024 * 1024):.2f} MB")

        return synthetic_data.copy(), {
            "execution_time_sec": end_time - start_time,
            "peak_memory_mb": peak_memory / (1024 * 1024),
        }
