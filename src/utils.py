import os
import numpy as np
import pandas as pd
import logging

def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'analysis.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def ensure_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def load_all_npy(folder_path, expected_features=None):
    """
    Loads .npy files and returns them as a single DataFrame.
    If expected_features is provided (as int or list), filters files accordingly.
    """
    dataframes = []
    logger = logging.getLogger(__name__)
    
    # Convert single int to list for consistency
    if isinstance(expected_features, int):
        expected_features = [expected_features]

    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            file_path = os.path.join(folder_path, file)
            data = np.load(file_path)
            num_features = data.shape[1]

            if expected_features is not None:
                if num_features not in expected_features:
                    logger.debug(f"Skipping {file}: {num_features} features (expected {expected_features})")
                    continue 

            df = pd.DataFrame(data)
            df["source_file"] = file
            df["feature_count"] = num_features
            dataframes.append(df)

    if not dataframes:
        return pd.DataFrame()

    return pd.concat(dataframes, ignore_index=True)
