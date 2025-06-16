
import pandas as pd
import time
import tracemalloc
from sdv.single_table import TVAESynthesizer
from sdv.metadata import Metadata
from pathlib import Path

class TVAESynthesizerWrapper:
    def __init__(self, output_dir: str = "data/synthetic/tvae"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fit_and_generate(self, df: pd.DataFrame, dataset_name: str):
        # Detect metadata
        print("Detecting metadata from input dataframe...")
        metadata = Metadata.detect_from_dataframe(data=df)




        # Init synthesizer
        synthesizer = TVAESynthesizer(metadata)

        # Track memory & time
        tracemalloc.start()
        start_time = time.time()

        # Fit model
        synthesizer.fit(df)
        # Generate synthetic data
        synthetic_data = synthesizer.sample(num_rows=len(df))


        end_time = time.time()
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()


        # Save to CSV
        out_path = self.output_dir / f"{dataset_name}_tvae.csv"
        synthetic_data.to_csv(out_path, index=False)

        print(f" Synthetic data saved to: {out_path}")
        print(f" Execution Time: {end_time - start_time:.2f} seconds")
        print(f" Peak Memory Usage: {peak_memory / (1024 * 1024):.2f} MB")

        return synthetic_data, {
            "execution_time_sec": end_time - start_time,
            "peak_memory_mb": peak_memory / (1024 * 1024)
        }
