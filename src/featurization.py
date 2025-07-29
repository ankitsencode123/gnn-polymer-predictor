def atom_features(atom):
    features = []
    atomic_numbers = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    features.extend([atom.GetAtomicNum() == x for x in atomic_numbers])
    features.extend([atom.GetDegree() == x for x in range(6)])
    features.extend([atom.GetFormalCharge() == x for x in range(-2, 3)])
    features.extend([atom.GetHybridization() == x for x in [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]])
    features.append(atom.GetIsAromatic())
    features.append(atom.IsInRing())
    return np.array(features, dtype=np.float32)

def bond_features(bond):
    features = []
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]
    features.extend([bond.GetBondType() == x for x in bond_types])
    features.append(bond.GetIsConjugated())
    features.append(bond.IsInRing())
    return np.array(features, dtype=np.float32)

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    atom_feat = []
    for atom in mol.GetAtoms():
        atom_feat.append(atom_features(atom))
    if len(atom_feat) == 0:
        return None
    x = torch.tensor(np.array(atom_feat), dtype=torch.float)
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])
        bond_feat = bond_features(bond)
        edge_attrs.extend([bond_feat, bond_feat])
    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_attrs), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
print("Testing featurization...")
test_smiles = "*CC(*)c1ccccc1C(=O)OCCCCCC"
test_graph = smiles_to_graph(test_smiles)
if test_graph is not None:
    print(f"Node features shape: {test_graph.x.shape}")
    print(f"Edge index shape: {test_graph.edge_index.shape}")
    print(f"Edge attributes shape: {test_graph.edge_attr.shape}")
else:
    print("Failed to create graph from test SMILES")

train_graphs = []
train_targets = []
valid_indices = []

print("Creating training graphs...")
for idx, row in train_df_clean.iterrows():
    graph = smiles_to_graph(row['SMILES'])
    if graph is not None:
        train_graphs.append(graph)
        targets = [row[col] for col in property_columns]
        train_targets.append(targets)
        valid_indices.append(idx)

print(f"Successfully created {len(train_graphs)} training graphs")
train_targets = np.array(train_targets, dtype=np.float32)
print(f"Training targets shape: {train_targets.shape}")

scaler = StandardScaler()
train_targets_scaled = scaler.fit_transform(train_targets)
print(f"Scaled targets shape: {train_targets_scaled.shape}")

train_idx, val_idx = train_test_split(
    range(len(train_graphs)),
    test_size=0.2,
    random_state=42,
    stratify=None
)

train_graphs_split = [train_graphs[i] for i in train_idx]
val_graphs_split = [train_graphs[i] for i in val_idx]
train_targets_split = train_targets_scaled[train_idx]
val_targets_split = train_targets_scaled[val_idx]

print(f"Train split: {len(train_graphs_split)} graphs")
print(f"Validation split: {len(val_graphs_split)} graphs")
print(f"Train targets split shape: {train_targets_split.shape}")
print(f"Val targets split shape: {val_targets_split.shape}")

for i, (graph, target) in enumerate(zip(train_graphs_split, train_targets_split)):
    graph.y = torch.tensor(target, dtype=torch.float).reshape(1, -1)

for i, (graph, target) in enumerate(zip(val_graphs_split, val_targets_split)):
    graph.y = torch.tensor(target, dtype=torch.float).reshape(1, -1)

print("Creating test graphs...")
test_graphs = []
test_valid_indices = []

for idx, row in test_df.iterrows():
    graph = smiles_to_graph(row['SMILES'])
    if graph is not None:
        test_graphs.append(graph)
        test_valid_indices.append(idx)

print(f"Successfully created {len(test_graphs)} test graphs")
