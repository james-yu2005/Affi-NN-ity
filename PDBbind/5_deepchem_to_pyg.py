def process_deepchem_dataset(dc_dataset):
    molecule_data = []

    for X, y, w, ids in train_dataset.iterbatches(batch_size=1, deterministic=True):
        ligand_path = X[0][0]
        print(ligand_path)
        protein_path = X[0][1]
        print(protein_path)

        ligand_mol = Chem.MolFromMolFile(str(ligand_path))

        if ligand_mol is None:
          print(f"Failed to load ligand molecule from {ligand_path}")
          continue  # skip this sample

        node_feats = get_node_features(ligand_mol)
        edge_feats = get_edge_features(ligand_mol)
        edge_index = get_adjacency_info(ligand_mol)

        protein_sequence = extract_sequence_from_pdb(protein_path)
        target_features = encode_sequence(protein_sequence, aa_to_idx)
        target_features = torch.tensor(target_features, dtype=torch.long).unsqueeze(0)

        data = Data(
            x=node_feats,
            edge_index=edge_index,
            edge_attr=edge_feats,
            y=torch.tensor(y, dtype=torch.float)
        )
        data.target_features = target_features

        molecule_data.append(data)

    return molecule_data
