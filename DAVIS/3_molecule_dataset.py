class MoleculeDataset(Dataset):

    # Important Parameter:
    # max_seq_length: Max number of amino acids to one-hot encode per protein (default is 1000)
    def __init__(self, root, dataframe, split='train', test_fraction=0.2, val_fraction=0.1,
                 transform=None, pre_transform=None, max_seq_length=1000, random_state=42):
        self.dataframe = dataframe.reset_index()

        # Following defines how the data will be split for train, test and val
        self.split = split
        self.test_fraction = test_fraction
        self.val_fraction = val_fraction
        self.random_state = random_state

        # Initialize empty list for processed data. This will later hold the fully processed dataset
        # Each item is a PyG Data object representing a drug-target pair
        self.molecule_data = []
        self.max_seq_length = max_seq_length

        # This method (defined next) actually splits the dataframe for train, test, val
        self._split_data()

        super(MoleculeDataset, self).__init__(root, transform, pre_transform)


    # Split the dataframe into train, validation and test sets
    def _split_data(self):
        from sklearn.model_selection import train_test_split

        # First split off the test set
        train_val_df, test_df = train_test_split(
            self.dataframe,
            test_size=self.test_fraction,
            random_state=self.random_state
        )

        # Then split the train set into train and validation
        if self.val_fraction > 0:
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=self.val_fraction / (1 - self.test_fraction),
                random_state=self.random_state
            )
        else:
            train_df = train_val_df
            val_df = train_val_df.iloc[0:0]  # Empty DataFrame with same columns

        # Assign the appropriate dataframe based on the split parameter
        if self.split == 'train':
            self.dataframe = train_df
        elif self.split == 'val':
            self.dataframe = val_df
        elif self.split == 'test':
            self.dataframe = test_df
        else:
            raise ValueError(f"Split '{self.split}' not recognized. Use 'train', 'val', or 'test'.")


    # Process molecules from SMILES into graph format and convert protein sequences to one-hot encoding
    def process(self):
        for index, row in tqdm(self.dataframe.iterrows(), total=self.dataframe.shape[0]): # Going row-by-row through the dataset, showing a progress bar with tqdm
            # Ensure column names are correct
            smiles = row["Drug"]  # If "Drug", it contains SMILES strings
            target_seq = row["Target"]  # If "Target", it contains protein sequences

            # Using RDKit to convert the SMILES string into a molecule object. If it fails (invalid SMILES), skip it.
            mol_obj = Chem.MolFromSmiles(smiles)
            if mol_obj is None:
                continue

            # Creates of tensor where each row represents a single atom in the molecule and its columns represents its features (like hybridization, aromatic ring, etc.)
            node_feats = self._get_node_features(mol_obj)

            # Goes through each bond in the molecule and extracts whether its single or not and whether its part of a ring or not.
            # Each bond is stored twice (once for each direction) so the tensor shape is [num_edges * 2, 2]
            edge_feats = self._get_edge_features(mol_obj)

            # Tensor shape: [2, num_edges]
            # First row is the source node (from where the bond originates). Second row is the destination node
            # NOT an adjacency matrix. Each column is an edge the first row number is the start and second row number is the end
            edge_index = self._get_adjacency_info(mol_obj)

            # One-hot encodes the amino acids (method defined next)
            target_features = self._sequence_to_one_hot(target_seq)

            # Data is a class that represents a single drug molecule graph
            data = Data(
                x=node_feats,
                edge_index=edge_index,
                edge_attr=edge_feats,
                y=torch.tensor([row["Y"]], dtype=torch.float)
            )

            # Attach the one-hot-encoded proteins to their respective drug graphs
            data.target_features = target_features

            # Store in list instead of saving to disk
            self.molecule_data.append(data)


    # Given a protein sequence, we convert it into a one-hot-encoding of shape [max_seq_length, 20]
    # Each row is one amino acid (up to 1000). Each column is one of the 20 standard amino acids
    def _sequence_to_one_hot(self, sequence):
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        encoding = np.zeros((self.max_seq_length, len(amino_acids)), dtype=np.float32)
        for idx, amino_acid in enumerate(sequence[:self.max_seq_length]):
            if amino_acid in amino_acids:
                encoding[idx, amino_acids.index(amino_acid)] = 1
        return torch.tensor(encoding.flatten(), dtype=torch.float).unsqueeze(0) # Here we flatten the tensor from [1000 * 20] to [20000]

    # ISSUE 07.05.25
    # If we return [20000], and batch 32 of them together, PyG tries to stack them as torch.cat([ [20,000], [20,000], ..., [20,000] ]) giving [640,000]
    # Instead of getting: [32, 20000] which is what the Linear layer expects
    # Unsqueeze adds a new first dimension converting [20,000] to [1, 20,000]
    # NOTE: [1, 20000] doesn't mean there's a physical "1" in the beginning of each vector.
    # The 1 is a new dimension that converts each flattened tensor into a row so that when 32 (batch size) such tensors are stacked it becomes a 2D matrix.
    # Without the 1, if 32 tensors were stacked it would become a really long 1D matrix giving [640,000] instead of [32, 64,000].


    # The following 3 functions have been (kind of) explained in the process function. Basically extracts node & edge features.
    def _get_node_features(self, mol):
        all_node_feats = []
        for atom in mol.GetAtoms():
            node_feats = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetHybridization(),
                atom.GetIsAromatic(),
                atom.GetTotalNumHs(),
                atom.GetNumRadicalElectrons(),
                atom.IsInRing(),
                atom.GetChiralTag()
            ]
            all_node_feats.append(node_feats)
        return torch.tensor(np.array(all_node_feats), dtype=torch.float)


    def _get_edge_features(self, mol):
        all_edge_feats = []
        for bond in mol.GetBonds():
            edge_feats = [
                bond.GetBondTypeAsDouble(),
                bond.IsInRing()
            ]
            all_edge_feats += [edge_feats, edge_feats]  # Bidirectional edges
        return torch.tensor(np.array(all_edge_feats), dtype=torch.float)


    def _get_adjacency_info(self, mol):
        edge_indices = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]  # Bidirectional edges
        return torch.tensor(edge_indices, dtype=torch.long).t().contiguous()


    def len(self):
        return len(self.dataframe)


    # Returns the processed molecule at index idx
    # Each item in self.molecule_data is a PyG Data object representing one drug–protein pair.
    # This is the empty list we initialized earlier
    def get(self, idx):
        return self.molecule_data[idx]


    # Return processed file names. Since we're storing in memory, we'll return an empty list or a dummy file name. """
    def processed_file_names(self):
        return []