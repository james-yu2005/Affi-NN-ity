import torch
from torch_geometric.loader import DataLoader
from src.models import GINModel
from src.engine import train_model, evaluate_model
from src.utils import plot_predictions_vs_actual
from data_loaders.hiqbind import load_pickle_dataset, process_hiqbind_dataset

if __name__ == "__main__":
    # CONFIG
    PICKLE_PATH = "path/to/hiqbind_dataset_clean.pkl" # UPDATE THIS PATH

    # 1. Load Data
    print("--- Loading HiQBind Data ---")
    try:
        train_dict, valid_dict, test_dict = load_pickle_dataset(PICKLE_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {PICKLE_PATH}. Run scripts/hiqbind_etl.py first.")
        exit()

    # 2. Featurize
    print("--- Featurizing ---")
    train_graphs = process_hiqbind_dataset(train_dict)
    valid_graphs = process_hiqbind_dataset(valid_dict)
    test_graphs = process_hiqbind_dataset(test_dict)

    # 3. Loaders
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(valid_graphs, batch_size=32)
    test_loader = DataLoader(test_graphs, batch_size=32)

    # 4. Train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training on {device} ---")
    model = GINModel().to(device)
    model, best_loss = train_model(model, train_loader, val_loader, device=device)

    # 5. Evaluate
    print("--- Evaluating on HiQBind Test ---")
    evaluate_model(model, test_loader, device)
    plot_predictions_vs_actual(model, test_loader, device, title="HiQBind Test Predictions")