# Each protein is one-hot encoded as a [1000, 20] matrix
seq_dim = 1000 * 20

class GINDrugTargetModel(torch.nn.Module):
    def __init__(self, node_feat_dim=9, seq_dim=20000, hidden_dim=128, output_dim=1):
        super(GINDrugTargetModel, self).__init__()

        # Node feature embedding
        self.node_embedding = Sequential(
            Linear(node_feat_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )

        # GIN convolution layers
        # Atoms exchanging information with their neighbors happens here
        nn1 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)

        nn2 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)

        self.seq_embedding = torch.nn.Sequential(
            torch.nn.Linear(seq_dim,   hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, hidden_dim//2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim//2, output_dim)
        )

    # Forward Pass
    def forward(self, batch):
      x = self.node_embedding(batch.x)
      x = F.relu(self.conv1(x, batch.edge_index))
      x = self.bn1(x)
      x = F.relu(self.conv2(x, batch.edge_index))
      x = self.bn2(x)

      # Only do this ONCE
      x = global_add_pool(x, batch.batch)

      seq = batch.target_features  # shape: [batch_size, seq_dim]
      seq_emb = self.seq_embedding(seq)  # shape: [batch_size, hidden_dim]

      combined = torch.cat([x, seq_emb], dim=1)  # shape: [batch_size, 2*hidden_dim]
      return self.predictor(combined)
