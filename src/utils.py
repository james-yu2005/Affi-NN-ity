import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_predictions_vs_actual(model, loader, device, title="Predicted vs Actual"):
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch).squeeze().cpu().numpy()
            targets = batch.y.squeeze().cpu().numpy()

            if preds.ndim == 0: preds = np.array([preds])
            if targets.ndim == 0: targets = np.array([targets])

            all_preds.extend(preds)
            all_targets.extend(targets)

    plt.figure(figsize=(8, 8))
    plt.scatter(all_targets, all_preds, alpha=0.5)
    plt.xlabel("Actual pKd")
    plt.ylabel("Predicted pKd")
    plt.title(title)
    plt.grid(True)

    min_val = min(min(all_targets), min(all_preds))
    max_val = max(max(all_targets), max(all_preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.show()