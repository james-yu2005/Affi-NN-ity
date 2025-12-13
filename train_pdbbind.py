import torch
from torch_geometric.loader import DataLoader
from src.models import GINModel
from src.engine import train_model, evaluate_model
from src.utils import plot_predictions_vs_actual
from data_loaders.pdbbind import load_pdbbind_datasets, process_deepchem_dataset

if __name__ == "__main__":
    # 1. Load Data
    print("--- Loading PDBBind Data ---")
    train_list, test_list = load_pdbbind_datasets()

    # 2. Featurize
    print("--- Featurizing ---")
    train_graphs = process_deepchem_dataset(train_list)
    test_graphs = process_deepchem_dataset(test_list)

    # 3. Create Loaders
    # Splitting Train into Train/Val (90/10)
    train_size = int(0.9 * len(train_graphs))
    valid_size = len(train_graphs) - train_size
    train_set, val_set = torch.utils.data.random_split(train_graphs, [train_size, valid_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)
    test_loader = DataLoader(test_graphs, batch_size=32)

    # 4. Train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training on {device} ---")
    model = GINModel().to(device)
    model, best_loss = train_model(model, train_loader, val_loader, device=device)

    # 5. Evaluate
    print("--- Evaluating on Core Set ---")
    evaluate_model(model, test_loader, device)
    plot_predictions_vs_actual(model, test_loader, device, title="PDBBind Core Set Predictions")