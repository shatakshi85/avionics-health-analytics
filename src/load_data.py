import numpy as np
import pandas as pd

def load_npy(path):
    data = np.load(path)
    df = pd.DataFrame(data)
    print("Loaded shape:", df.shape)
    return df
