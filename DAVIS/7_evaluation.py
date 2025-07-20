def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            preds.append(pred.view(-1).cpu())
            trues.append(batch.y.view(-1).cpu())

    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)

    mse = F.mse_loss(preds, trues)
    print(f'Test MSE: {mse.item():.4f}')

    # Print few predictions vs actual value
    print("\nSample Predictions vs True Binding Affinities:")
    for i in range(min(100, len(preds))):  # Show 10 samples (or fewer if smaller dataset)
        print(f"True: {trues[i].item():.4f}, Predicted: {preds[i].item():.4f}")

    return preds, trues, mse.item()

# Test the model
test_preds, test_targets, test_mse = evaluate_model(trained_model, test_loader)