# Create train, validation, and test datasets. The process() method converts each row into a PyTorch Geometric Data object
train_dataset = MoleculeDataset(root='.', dataframe=df_DAVIS, split='train')
train_dataset.process()

val_dataset = MoleculeDataset(root='.', dataframe=df_DAVIS, split='val')
val_dataset.process()

test_dataset = MoleculeDataset(root='.', dataframe=df_DAVIS, split='test')
test_dataset.process()

# Create DataLoaders
from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)