# prompt: generate a segment for relevant evaluation metrics

import numpy as np
from sklearn.metrics import r2_score

def evaluate_model(model, loader, device='cpu'):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch).squeeze().cpu().numpy()
            targets = batch.y.squeeze().cpu().numpy()

            # Handle cases where preds or targets might be single values
            if preds.ndim == 0:
                preds = np.array([preds])
            if targets.ndim == 0:
                targets = np.array([targets])

            all_preds.extend(preds)
            all_targets.extend(targets)

    mse = mean_squared_error(all_targets, all_preds)
    rmse = math.sqrt(mse)
    r2 = r2_score(all_targets, all_preds)

    print(f"Evaluation Results:")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  R^2 Score: {r2:.4f}")


evaluate_model(debug_model, test_loader, device='cpu')