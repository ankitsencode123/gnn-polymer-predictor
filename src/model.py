class MultiTaskGNN(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim=256, num_layers=4, dropout=0.2):
        super(MultiTaskGNN, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        self.edge_embedding = nn.Linear(edge_features, hidden_dim)
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.global_pool = global_mean_pool
        self.task_heads = nn.ModuleList()
        for i in range(5):
            head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 4, 1)
            )
            self.task_heads.append(head)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if x.size(1) != self.node_features:
            raise ValueError(f"Expected node features: {self.node_features}, got: {x.size(1)}")
        x = self.node_embedding(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            x_residual = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i > 0:
                x = x + x_residual
        x = self.global_pool(x, batch)
        outputs = []
        for head in self.task_heads:
            out = head(x)
            outputs.append(out)
        return torch.cat(outputs, dim=1)

if len(train_graphs_split) > 0:
    sample_graph = train_graphs_split[0]
    node_features = sample_graph.x.shape[1]
    edge_features = sample_graph.edge_attr.shape[1] if sample_graph.edge_attr.size(0) > 0 else 6
    print(f"Node features dimension: {node_features}")
    print(f"Edge features dimension: {edge_features}")
    model = MultiTaskGNN(
        node_features=node_features,
        edge_features=edge_features,
        hidden_dim=256,
        num_layers=4,
        dropout=0.2
    )
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Model: {model}")
else:
    print("No training graphs available for model creation")
    
class WeightedMAELoss(nn.Module):
    def __init__(self, train_targets, K=5):
        super(WeightedMAELoss, self).__init__()
        self.K = K
        self.weights = self.calculate_weights(train_targets)
        print(f"Calculated weights: {self.weights}")

    def calculate_weights(self, targets):
        targets = targets.copy()
        n_samples, n_properties = targets.shape
        weights = []
        for i in range(n_properties):
            property_values = targets[:, i]
            valid_mask = ~np.isnan(property_values)
            valid_values = property_values[valid_mask]
            if len(valid_values) == 0:
                weights.append(1.0)
                continue
            n_i = len(valid_values)
            r_i = np.max(valid_values) - np.min(valid_values)
            if r_i == 0:
                r_i = 1.0
            scale_factor = 1.0 / r_i
            inverse_sqrt_factor = np.sqrt(1.0 / n_i)
            weight = scale_factor * inverse_sqrt_factor
            weights.append(weight)
        weights = np.array(weights)
        sum_weights = np.sum(weights)
        if sum_weights > 0:
            weights = (weights / sum_weights) * self.K
        return torch.tensor(weights, dtype=torch.float32)

    def forward(self, predictions, targets):
        if predictions.size() != targets.size():
            raise ValueError(f"Size mismatch: predictions {predictions.size()}, targets {targets.size()}")
        weights = self.weights.to(predictions.device)
        abs_errors = torch.abs(predictions - targets)
        weighted_errors = abs_errors * weights.unsqueeze(0)
        mae_per_property = torch.mean(weighted_errors, dim=0)
        wmae = torch.mean(mae_per_property)
        return wmae
train_targets_for_weights = train_targets.copy()
for i in range(train_targets_for_weights.shape[1]):
    col_mean = np.nanmean(train_targets_for_weights[:, i])
    train_targets_for_weights[:, i] = np.where(
        np.isnan(train_targets_for_weights[:, i]),
        col_mean,
        train_targets_for_weights[:, i]
    )
criterion = WeightedMAELoss(train_targets_for_weights)
print("Weighted MAE Loss function created successfully")

evice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = model.to(device)
criterion = criterion.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

train_loader = DataLoader(train_graphs_split, batch_size=32, shuffle=True)
val_loader = DataLoader(val_graphs_split, batch_size=32, shuffle=False)

print(f"Train loader: {len(train_loader)} batches")
print(f"Val loader: {len(val_loader)} batches")
