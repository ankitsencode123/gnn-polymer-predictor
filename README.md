# GNN Polymer Property Predictor

A Graph Neural Network (GNN) based machine learning model for predicting multiple polymer properties from SMILES molecular representations. This project implements a multi-task learning approach to simultaneously predict five key polymer properties: Glass Transition Temperature (Tg), Fractional Free Volume (FFV), Critical Temperature (Tc), Density, and Radius of Gyration (Rg).

## 🎯 Overview

This repository contains a complete pipeline for:
- Converting SMILES strings to molecular graphs
- Training a multi-task Graph Convolutional Network (GCN)
- Predicting polymer properties with weighted loss optimization
- Handling missing data through advanced imputation techniques

## 🔬 Model Architecture

The model uses a **Multi-Task Graph Neural Network** with the following components:

- **Graph Convolution Layers**: 4-layer GCN with residual connections
- **Node Features**: 28-dimensional atom feature vectors (atomic number, degree, formal charge, hybridization, aromaticity, ring membership)
- **Edge Features**: 6-dimensional bond feature vectors (bond type, conjugation, ring membership)
- **Multi-Task Heads**: Separate prediction heads for each property
- **Global Pooling**: Mean pooling for graph-level representations

### Key Features

- **Weighted MAE Loss**: Custom loss function that accounts for property-specific scales and data availability
- **Data Imputation**: KNN + Iterative imputation for handling missing values
- **Regularization**: Dropout, batch normalization, and gradient clipping
- **Early Stopping**: Prevents overfitting with patience-based stopping

## 📊 Dataset

The model expects training and test datasets with the following structure:

| Column | Description |
|--------|-------------|
| id | Unique identifier |
| SMILES | Molecular structure in SMILES format |
| Tg | Glass Transition Temperature |
| FFV | Fractional Free Volume |
| Tc | Critical Temperature |
| Density | Polymer density |
| Rg | Radius of Gyration |

## 🚀 Installation

### Requirements

```bash
pip install rdkit
pip install torch_geometric
pip install pandas numpy torch scikit-learn
```

### Dependencies

- **RDKit**: For molecular graph processing
- **PyTorch Geometric**: For graph neural network operations
- **PyTorch**: Deep learning framework
- **Scikit-learn**: Data preprocessing and imputation
- **Pandas/NumPy**: Data manipulation

## 💻 Usage

### Basic Usage

```python
import pandas as pd
from src.model import MultiTaskGNN
from src.data_processing import smiles_to_graph, process_data

# Load your data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Process and train
model = MultiTaskGNN(node_features=28, edge_features=6)
# ... training code ...

# Make predictions
predictions = model.predict(test_graphs)
```

### Training Process

1. **Data Preprocessing**:
   - Remove samples with all missing target values
   - Apply KNN imputation followed by iterative imputation
   - Standardize target variables

2. **Graph Construction**:
   - Convert SMILES to molecular graphs using RDKit
   - Extract atom and bond features
   - Create PyTorch Geometric Data objects

3. **Model Training**:
   - Multi-task learning with weighted MAE loss
   - AdamW optimizer with learning rate scheduling
   - Early stopping based on validation loss

4. **Prediction**:
   - Generate predictions for test molecules
   - Inverse transform to original scale
   - Handle invalid SMILES with mean imputation

## 📈 Performance

The model achieves competitive performance on polymer property prediction:

- **Training Loss**: ~0.52 (final epoch)
- **Validation Loss**: ~0.49 (best model)
- **Architecture**: 480K+ parameters
- **Training Time**: ~100 epochs with early stopping

### Property-Specific Weights

The weighted loss function automatically balances learning across properties:
- Accounts for different scales and units
- Adjusts for data availability per property
- Prevents bias toward properties with larger numerical ranges

## 🔧 Configuration

Key hyperparameters that can be tuned:

```python
MODEL_CONFIG = {
    'hidden_dim': 256,
    'num_layers': 4,
    'dropout': 0.2,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'batch_size': 32,
    'max_epochs': 100,
    'patience': 15
}
```



## 🧪 Key Innovations

1. **Multi-Task Learning**: Simultaneous prediction of multiple correlated properties
2. **Weighted Loss Function**: Automatic balancing of property-specific learning
3. **Advanced Imputation**: Sequential KNN and iterative imputation
4. **Robust Graph Features**: Comprehensive atom and bond feature engineering
5. **Residual Connections**: Improved gradient flow in deep GCN architecture

## 🔍 Technical Details

### Atom Features (28D)
- Atomic number (one-hot encoded for common atoms)
- Degree (0-5)
- Formal charge (-2 to +2)
- Hybridization (SP, SP2, SP3, SP3D, SP3D2)
- Aromaticity and ring membership flags

### Bond Features (6D)
- Bond type (single, double, triple, aromatic)
- Conjugation and ring membership flags

### Loss Function
```
WMAE = (1/K) * Σ(w_i * |y_i - ŷ_i|)
```
Where weights w_i are calculated based on property range and data availability.

## 📊 Results Analysis

The trained model shows:
- Consistent convergence over 90+ epochs
- Good generalization (train-val loss gap < 0.03)
- Reasonable predictions across all property types
- Robust handling of diverse polymer structures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **RDKit**: For molecular informatics tools
- **PyTorch Geometric**: For graph neural network framework
- **Graph Neural Networks**: Inspiration from molecular property prediction literature

---

**Note**: This model is designed for research and educational purposes. For production use, please validate predictions against experimental data and consider domain-specific requirements.
