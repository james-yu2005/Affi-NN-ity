import torch
import torch.nn.functional as F

def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            preds = model(batch).squeeze()  # Model predicts pKd scores
            targets = batch.y.squeeze()     # True pKd values from the data

            loss = F.mse_loss(preds, targets) # Compute mean squared error loss
            loss.backward() # Backpropagate the error
            optimizer.step() # Update weights

            total_train_loss += loss.item() * batch.num_graphs # batch.num_graphs is the number of samples in the batch (32)

        avg_train_loss = total_train_loss / len(train_loader.dataset) # This gives average MSE loss over all training samples.

        # Validation Phase
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                preds = model(batch).squeeze()
                targets = batch.y.squeeze()

                loss = F.mse_loss(preds, targets)
                total_val_loss += loss.item() * batch.num_graphs

        avg_val_loss = total_val_loss / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return model

from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
seq_dim = 1000 * 20
model = GINDrugTargetModel(node_feat_dim=9, seq_dim=seq_dim)

# Train the model
trained_model = train_model(model, train_loader, val_loader, num_epochs=40)