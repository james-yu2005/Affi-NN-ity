import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, BatchNorm1d, Dropout, LayerNorm
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool

class GINModel(torch.nn.Module):
    def __init__(self, node_feat_dim=9, protein_dim=640, fp_dim=1024, hidden_dim=128, output_dim=1, dropout=0.2):
        super(GINModel, self).__init__()

        # Node features embedding
        self.node_embedding = Sequential(
            Linear(node_feat_dim, hidden_dim),
            LayerNorm(hidden_dim),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, hidden_dim)
        )

        # GIN Layers
        self.conv1 = GINConv(Sequential(
            Linear(hidden_dim, hidden_dim), ReLU(), Dropout(dropout), Linear(hidden_dim, hidden_dim)
        ))
        self.bn1 = BatchNorm1d(hidden_dim)

        self.conv2 = GINConv(Sequential(
            Linear(hidden_dim, hidden_dim), ReLU(), Dropout(dropout), Linear(hidden_dim, hidden_dim)
        ))
        self.bn2 = BatchNorm1d(hidden_dim)

        # Encoders
        self.pocket_embedding = Sequential(
            Linear(protein_dim, hidden_dim * 2), LayerNorm(hidden_dim * 2), ReLU(), Dropout(dropout),
            Linear(hidden_dim * 2, hidden_dim), ReLU(), Dropout(dropout), Linear(hidden_dim, hidden_dim)
        )

        self.fp_embedding = Sequential(
            Linear(fp_dim, hidden_dim * 2), LayerNorm(hidden_dim * 2), ReLU(), Dropout(dropout),
            Linear(hidden_dim * 2, hidden_dim), ReLU(), Dropout(dropout), Linear(hidden_dim, hidden_dim)
        )

        # Attention
        self.attention = Sequential(
            Linear(3 * hidden_dim, hidden_dim), ReLU(), Dropout(dropout),
            Linear(hidden_dim, 3), torch.nn.Softmax(dim=1)
        )

        # Predictor
        self.predictor = Sequential(
            Linear(3 * hidden_dim, hidden_dim * 2), LayerNorm(hidden_dim * 2), ReLU(), Dropout(dropout),
            Linear(hidden_dim * 2, hidden_dim), ReLU(), Dropout(dropout),
            Linear(hidden_dim, hidden_dim // 2), ReLU(), Dropout(dropout * 0.5),
            Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, batch):
        x = self.node_embedding(batch.x)
        x1 = self.bn1(F.relu(self.conv1(x, batch.edge_index)))
        x2 = self.bn2(F.relu(self.conv2(x1, batch.edge_index)))
        x2 = x2 + x1  # Residual

        mol_emb = (global_add_pool(x2, batch.batch) + global_mean_pool(x2, batch.batch)) / 2
        pocket_emb = self.pocket_embedding(batch.target_features.float())
        fp_emb = self.fp_embedding(batch.morgan_fp.float())

        combined = torch.cat([mol_emb, pocket_emb, fp_emb], dim=1)
        weights = self.attention(combined)
        stacked = torch.stack([mol_emb, pocket_emb, fp_emb], dim=2)
        attended = (stacked * weights.unsqueeze(1)).sum(dim=2)
        
        # Combine original features with attended features
        final_features = torch.cat([combined, attended], dim=1)
        

        return self.predictor(combined)