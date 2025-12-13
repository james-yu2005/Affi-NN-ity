import pickle
import os

def load_pickle_dataset(pickle_path):
    print(f"Loading HiQBind pickle from {pickle_path}...")
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
        
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    
    return data['train'], data['valid'], data['test']