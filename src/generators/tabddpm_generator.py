import os
import time
import tracemalloc
import pandas as pd
import subprocess
import json
from pathlib import Path
from src.utils.postprocess import match_format

class TabDDPMGenerator:
    def __init__(self, output_dir: str = "data/synthetic/tabddpm", num_experiments: int = 1):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_experiments = num_experiments
        self.tabddpm_dir = Path("tabddpm")
        
    def fit_and_generate(self, df: pd.DataFrame, dataset_name: str, tabddpm_config: dict):
        """
        Generate synthetic data using TabDDPM
        
        Args:
            df: Input DataFrame
            dataset_name: Name of the dataset
            tabddpm_config: Configuration dictionary for TabDDPM
                - train_size: Number of training samples
                - eval_model: 'catboost' or 'mlp'
                - exp_name: Experiment name
                - n_sample_seeds: Number of sampling seeds
                - n_eval_seeds: Number of evaluation seeds
                - target_column: Name of the target column (default: last column)
                - categorical_columns: List of categorical column names
        """
        print("Initializing TabDDPM synthesizer...")
        
        # Ensure TabDDPM directory exists
        if not self.tabddpm_dir.exists():
            raise FileNotFoundError(f"TabDDPM directory not found at {self.tabddpm_dir}")
        
        # Preprocess data for TabDDPM format
        self._prepare_tabddpm_data(df, dataset_name, tabddpm_config)
        
        log = []
        for i in range(self.num_experiments):
            print(f"Running TabDDPM experiment {i+1}/{self.num_experiments}...")
            
            start_time = time.time()
            tracemalloc.start()
            
            # Run TabDDPM training and generation
            synthetic_data = self._run_tabddpm(dataset_name, tabddpm_config, i, df)
            
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            end_time = time.time()
            
            # Post-process synthetic data to match original format
            # Note: TabDDPM generates data in a different format, so we return it as-is
            # synthetic_data = match_format(synthetic_data, df)
            
            # Save synthetic data
            file_path = self.output_dir / f"{dataset_name}_tabddpm_{i}.csv"
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
    
    def _run_tabddpm(self, dataset_name: str, config: dict, experiment_idx: int, original_df: pd.DataFrame):
        """
        Run TabDDPM training and generation
        
        Args:
            dataset_name: Name of the dataset
            config: TabDDPM configuration
            experiment_idx: Current experiment index
            
        Returns:
            pandas.DataFrame: Generated synthetic data
        """
        # Create experiment directory
        exp_dir = self.tabddpm_dir / "exp" / dataset_name / f"tabddpm_exp_{experiment_idx}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Use a pre-existing config file instead of creating one
        config_file = exp_dir / "config.toml"
        print(f"Config file path: {config_file}")
        print(f"Config file exists: {config_file.exists()}")
        
        if not config_file.exists():
            # Copy a working config file
            import shutil
            base_config = self.tabddpm_dir / "exp" / "wilt" / "ddpm_cb_best" / "config.toml"
            print(f"Base config path: {base_config}")
            print(f"Base config exists: {base_config.exists()}")
            
            shutil.copy(base_config, config_file)
            
            # Update the config file for our dataset
            with open(config_file, 'r') as f:
                content = f.read()
            
            content = content.replace('parent_dir = "exp/wilt/ddpm_cb_best"', f'parent_dir = "exp/{dataset_name}/tabddpm_exp_{experiment_idx}"')
            content = content.replace('real_data_path = "data/wilt/"', f'real_data_path = "data/{dataset_name}"')
            content = content.replace('num_numerical_features = 5', f'num_numerical_features = {config.get("n_num_features", 6)}')
            content = content.replace('device = "cuda:0"', 'device = "cpu"')
            
            with open(config_file, 'w') as f:
                f.write(content)
            
            print(f"Config file created and updated")
        
        print(f"About to run TabDDPM pipeline...")
        
        # Run TabDDPM pipeline directly (skip tuning for now)
        pipeline_cmd = [
            "python", "scripts/pipeline.py",
            "--config", f"exp/{dataset_name}/tabddpm_exp_{experiment_idx}/config.toml",
            "--train", "--sample"
        ]
        
        print(f"Pipeline command: {pipeline_cmd}")
        print(f"Running TabDDPM pipeline: {' '.join(pipeline_cmd)}")
        print(f"Working directory: {self.tabddpm_dir}")
        print(f"Current directory: {os.getcwd()}")
        
        try:
            result = subprocess.run(
                pipeline_cmd,
                cwd=self.tabddpm_dir,
                capture_output=True,
                text=True
            )
            
            print(f"Pipeline return code: {result.returncode}")
            if result.stdout:
                print(f"Pipeline stdout: {result.stdout}")
            if result.stderr:
                print(f"Pipeline stderr: {result.stderr}")
            
            if result.returncode != 0:
                print(f"TabDDPM pipeline failed: {result.stderr}")
                raise RuntimeError(f"TabDDPM pipeline failed with return code {result.returncode}")
        except Exception as e:
            print(f"Exception during pipeline execution: {e}")
            raise
        
        # Load generated synthetic data
        synthetic_file = exp_dir / "synthetic.csv"
        if synthetic_file.exists():
            # Load existing CSV file
            synthetic_df = pd.read_csv(synthetic_file)
            print(f"Loaded synthetic data shape: {synthetic_df.shape}")
            print(f"Synthetic data columns: {list(synthetic_df.columns)}")
            
            # Map back to original column names and values
            synthetic_df = self._map_to_original_format(synthetic_df, original_df, config)
            if synthetic_df is None:
                print("Warning: Mapping failed, returning synthetic data as-is")
                return pd.read_csv(synthetic_file)
            return synthetic_df
        else:
            raise FileNotFoundError(f"Synthetic data file not found: {synthetic_file}")
    
    def _map_to_original_format(self, synthetic_df: pd.DataFrame, original_df: pd.DataFrame, config: dict):
        """
        Map synthetic data back to original column names and value formats
        
        Args:
            synthetic_df: Generated synthetic data
            original_df: Original input data
            config: Configuration with preprocessing info
            
        Returns:
            pd.DataFrame: Synthetic data with original column names and formats
        """
        import numpy as np
        
        print(f"Starting _map_to_original_format")
        print(f"Synthetic data shape: {synthetic_df.shape}")
        print(f"Config keys: {list(config.keys())}")
        
        if '_preprocessing_info' not in config:
            print("Warning: No preprocessing info found, returning synthetic data as-is")
            return synthetic_df
        
        preprocessing_info = config['_preprocessing_info']
        numerical_columns = preprocessing_info['numerical_columns']
        categorical_columns = preprocessing_info['categorical_columns']
        target_column = preprocessing_info['target_column']
        label_encoders = preprocessing_info['label_encoders']
        target_encoder = preprocessing_info['target_encoder']
        original_dtypes = preprocessing_info['original_dtypes']
        
        # Determine the structure of synthetic data
        n_num_features = len(numerical_columns)
        n_cat_features = len(categorical_columns)
        
        # Map synthetic data columns to original structure
        result_df = pd.DataFrame()
        
        # Map numerical features
        if n_num_features > 0:
            for i, col_name in enumerate(numerical_columns):
                if i < synthetic_df.shape[1]:
                    result_df[col_name] = synthetic_df.iloc[:, i]
        
        # Map categorical features
        if n_cat_features > 0:
            for i, col_name in enumerate(categorical_columns):
                if n_num_features + i < synthetic_df.shape[1]:
                    # Inverse transform categorical values
                    if col_name in label_encoders:
                        encoded_values = synthetic_df.iloc[:, n_num_features + i].astype(int)
                        # Handle out-of-range values
                        max_val = len(label_encoders[col_name].classes_) - 1
                        encoded_values = np.clip(encoded_values, 0, max_val)
                        result_df[col_name] = label_encoders[col_name].inverse_transform(encoded_values)
                    else:
                        result_df[col_name] = synthetic_df.iloc[:, n_num_features + i]
        
        # Map target column
        target_idx = n_num_features + n_cat_features
        if target_idx < synthetic_df.shape[1]:
            if target_encoder is not None:
                # Inverse transform target values
                encoded_target = synthetic_df.iloc[:, target_idx].astype(int)
                max_val = len(target_encoder.classes_) - 1
                encoded_target = np.clip(encoded_target, 0, max_val)
                result_df[target_column] = target_encoder.inverse_transform(encoded_target)
            else:
                result_df[target_column] = synthetic_df.iloc[:, target_idx]
        
        # Restore original data types
        for col in result_df.columns:
            if col in original_dtypes:
                try:
                    result_df[col] = result_df[col].astype(original_dtypes[col])
                except:
                    # If conversion fails, keep as is
                    pass
        
        return result_df
    
    def _create_config_file(self, config_file: Path, config: dict, dataset_name: str):
        """Create TabDDPM configuration file"""
        config_content = f"""
# TabDDPM Configuration
parent_dir = "{config_file.parent}"
real_data_path = "data/{dataset_name}"
model_type = "mlp"
num_numerical_features = {config.get('n_num_features', 6)}

[train.main]
lr = {config.get("lr", 0.0002)}
weight_decay = {config.get("weight_decay", 1e-4)}
batch_size = {config.get("batch_size", 1024)}
steps = {config.get("num_epochs", 100)}

[train.T]
normalization = "quantile"
cat_encoding = "ordinal"
cat_nan_policy = null
cat_min_frequency = 0.01

[model_params]
is_y_cond = false
num_classes = {config.get("num_classes", 2)}

[model_params.rtdl_params]
d_layers = {config.get("hidden_dims", [256, 256])}
dropout = {config.get("dropout", 0.1)}

[diffusion_params]
gaussian_loss_type = "mse"
num_timesteps = {config.get("num_timesteps", 1000)}
scheduler = "cosine"

[sample]
num_samples = {config.get("num_samples", 1000)}
batch_size = {config.get("batch_size", 1024)}
seed = 0

[eval.type]
eval_type = "synthetic"
eval_model = "{config.get('eval_model', 'catboost')}"

[eval.T]
normalization = "quantile"
cat_encoding = "ordinal"
cat_nan_policy = null
cat_min_frequency = 0.01

seed = 0
device = "cpu"
"""
        
        with open(config_file, 'w') as f:
            f.write(config_content)
    
    def _prepare_tabddpm_data(self, df: pd.DataFrame, dataset_name: str, config: dict):
        """
        Prepare data in TabDDPM format
        
        Args:
            df: Input DataFrame
            dataset_name: Name of the dataset
            config: Configuration dictionary
        """
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        
        # Determine target column
        target_column = config.get('target_column', df.columns[-1])
        if target_column not in df.columns:
            target_column = df.columns[-1]
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical columns
        categorical_columns = config.get('categorical_columns', [])
        numerical_columns = [col for col in X.columns if col not in categorical_columns]
        
        # Encode categorical variables
        X_encoded = X.copy()
        label_encoders = {}
        
        for col in categorical_columns:
            if col in X_encoded.columns:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                label_encoders[col] = le
        
        # Encode target if categorical
        if y.dtype == 'object' or y.dtype.name == 'category':
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))
            num_classes = len(target_encoder.classes_)
            task_type = 'binclass' if num_classes == 2 else 'multiclass'
        else:
            # Check if it's binary classification (0/1 values)
            unique_values = y.unique()
            if len(unique_values) == 2 and set(unique_values) == {0, 1}:
                num_classes = 2
                task_type = 'binclass'
            else:
                num_classes = 0
                task_type = 'regression'
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_encoded, y, test_size=0.3, random_state=42, stratify=y if task_type != 'regression' else None
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp if task_type != 'regression' else None
        )
        
        # Create data directory
        data_dir = self.tabddpm_dir / "data" / dataset_name
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save numerical features
        if len(numerical_columns) > 0:
            X_num_train = X_train[numerical_columns].values.astype(np.float32)
            X_num_val = X_val[numerical_columns].values.astype(np.float32)
            X_num_test = X_test[numerical_columns].values.astype(np.float32)
            
            np.save(data_dir / "X_num_train.npy", X_num_train)
            np.save(data_dir / "X_num_val.npy", X_num_val)
            np.save(data_dir / "X_num_test.npy", X_num_test)
        else:
            # Create empty files if no numerical features
            empty_array = np.array([]).reshape(0, 0)
            np.save(data_dir / "X_num_train.npy", empty_array)
            np.save(data_dir / "X_num_val.npy", empty_array)
            np.save(data_dir / "X_num_test.npy", empty_array)
        
        # Save categorical features
        if len(categorical_columns) > 0:
            X_cat_train = X_train[categorical_columns].values.astype(np.int32)
            X_cat_val = X_val[categorical_columns].values.astype(np.int32)
            X_cat_test = X_test[categorical_columns].values.astype(np.int32)
            
            np.save(data_dir / "X_cat_train.npy", X_cat_train)
            np.save(data_dir / "X_cat_val.npy", X_cat_val)
            np.save(data_dir / "X_cat_test.npy", X_cat_test)
        
        # Save target
        np.save(data_dir / "y_train.npy", y_train.values if hasattr(y_train, 'values') else y_train)
        np.save(data_dir / "y_val.npy", y_val.values if hasattr(y_val, 'values') else y_val)
        np.save(data_dir / "y_test.npy", y_test.values if hasattr(y_test, 'values') else y_test)
        
        # Create info.json
        info = {
            "task_type": task_type,
            "name": dataset_name,
            "id": f"{dataset_name}--id",
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "n_num_features": len(numerical_columns),
            "n_cat_features": len(categorical_columns)
        }
        
        if num_classes > 0:
            info["n_classes"] = num_classes
        
        import json
        with open(data_dir / "info.json", 'w') as f:
            json.dump(info, f, indent=4)
        
        print(f"Preprocessed data saved to {data_dir}")
        print(f"Task type: {task_type}, Numerical features: {len(numerical_columns)}, Categorical features: {len(categorical_columns)}")
        
        # Store the number of numerical features for config
        config['n_num_features'] = len(numerical_columns)
        config['num_classes'] = num_classes
        
        # Store preprocessing information for later mapping
        config['_preprocessing_info'] = {
            'numerical_columns': numerical_columns,
            'categorical_columns': categorical_columns,
            'target_column': target_column,
            'label_encoders': label_encoders,
            'target_encoder': target_encoder if 'target_encoder' in locals() else None,
            'original_dtypes': df.dtypes.to_dict()
        }
