import pandas as pd
import numpy as np


def match_format(synth_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    # Ensure column order
    synth_df = synth_df[reference_df.columns.tolist()]

    # Match dtypes and round floats
    for col in reference_df.columns:
        ref_dtype = reference_df[col].dtype
        if pd.api.types.is_float_dtype(ref_dtype):
            # Estimate decimal places in the reference
            decimals = reference_df[col].apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
            most_common_decimals = decimals.mode().iloc[0] if not decimals.empty else 2
            synth_df[col] = pd.to_numeric(synth_df[col], errors='coerce').round(most_common_decimals)
        elif pd.api.types.is_integer_dtype(ref_dtype):
            synth_df[col] = pd.to_numeric(synth_df[col], errors='coerce').round(0).astype('Int64')
        elif pd.api.types.is_categorical_dtype(ref_dtype):
            synth_df[col] = synth_df[col].astype('category')
        elif pd.api.types.is_bool_dtype(ref_dtype):
            synth_df[col] = synth_df[col].astype(bool)
        else:
            synth_df[col] = synth_df[col].astype(str)

    return synth_df
