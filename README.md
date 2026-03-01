# PolyketideClassifier

A Graph Neural Network classifier for distinguishing Polyketide Synthase (PKS) products from non-PKS molecules.

## Scientific Motivation

**Polyketides** are a large class of secondary metabolites produced by bacteria, fungi, and plants. They include many pharmaceutically important compounds (antibiotics, immunosuppressants, anticancer agents). PKS products have characteristic structural features arising from their biosynthetic assembly-line mechanism.

**The Classification Problem**: Given a molecule's structure, can we determine whether it was (or could be) produced by a PKS? This is challenging because:
1. PKS products are structurally diverse
2. Many non-PKS molecules share substructural features with PKS products
3. Standard molecular fingerprints (ECFP4) capture local substructures but may miss the global structural patterns characteristic of PKS biosynthesis

**Our Approach**: Train a Graph Attention Network (GAT) classifier directly on molecular graphs, learning to distinguish PKS products from hard negative non-PKS molecules generated via enzymatic and synthetic chemistry simulations.

## Quick Start

Run single-molecule inference with the pre-trained model:

```bash
python scripts/10_run_inference.py --smiles "CC(O)CC(=O)O"
```

This loads the trained checkpoint from `models/supervised_gnn/best_model.pt` and outputs a PKS probability.

## Data Pipeline

### Training Data Generation (Scripts 01-03)

```
01_generate_PKS_products.py          # Generate PKS products using the bcs library
02_generate_PKS_augmentations.py     # Generate hard-negative triplets (PKS + enzymatic + synthetic)
03_create_train_val_test_splits.py   # Connected-component train/val/test splits (no SMILES leakage)
```

For each PKS molecule, we generate a **triplet**:
1. **Anchor (PKS)**: The original PKS product (label = 1)
2. **Enzymatic augmentation**: Most similar DORAnet enzymatic product (label = 0)
3. **Synthetic augmentation**: Most similar DORAnet synthetic product (label = 0)

Hard negatives force the model to learn subtle structural differences rather than obvious chemical dissimilarities. Connected-component splitting prevents SMILES leakage across splits.

### OOD Evaluation Set Generation (Scripts 04-06)

```
04_generate_extender_ood_eval_set.py         # PKS from held-out extender codes
05_generate_mixed_extender_ood_set.py        # 2-extension PKS with 1 training + 1 OOD extender
06_generate_methyltransferase_eval_set.py    # Alpha-methylated PKS (simulating C-methyltransferase)
```

## Training

Train the supervised GNN classifier with distributed data parallel:

```bash
# Multi-GPU
torchrun --nproc_per_node=4 scripts/07_train_gnn_classifier.py

# Single GPU
python scripts/07_train_gnn_classifier.py

# Multi-node (SLURM)
srun --nodes=4 --ntasks-per-node=4 --gpus-per-node=4 \
    python scripts/07_train_gnn_classifier.py
```

Key hyperparameters (configurable via CLI args):
- `--batch_size 64` (per-GPU)
- `--epochs 30`
- `--lr 3e-4`
- `--pos_weight 2.0` (BCEWithLogitsLoss class weighting)
- `--scheduler` (optional cosine annealing with warmup)

## Evaluation

```bash
# Test set evaluation (GNN vs ECFP4 baseline)
python scripts/08_test_gnn_classifier.py

# OOD recall evaluation
python scripts/09_evaluate_ood_recall.py
```

### Results

**In-distribution (test set):**

| Method | AUPRC | Accuracy |
|--------|-------|----------|
| Supervised GNN | 0.999+ | 99.7% |
| ECFP4 LogReg | 0.999+ | 99.7% |

**Out-of-distribution (extender-code OOD, 678 molecules):**

| Method | OOD Recall |
|--------|-----------|
| Supervised GNN (best checkpoint) | ~46% |
| ECFP4 LogReg | ~46% |

**Methyltransferase-modified eval sets:**

| Eval Set | GNN Recall | ECFP4 Recall |
|----------|-----------|-------------|
| Training extenders, 1-2 ext + methyl | 99.9% | 6.5% |
| Mixed (1 train + 1 OOD ext) + methyl | 45.3% | 4.5% |

The GNN and ECFP4 have complementary failure modes: the GNN struggles with novel extender codes, while ECFP4 is fragile to structural modifications like alpha-methylation.

## Architecture

The classifier uses a GAT-based architecture with:
- **Graph construction**: On-the-fly from SMILES via RDKit
- **Atom features**: One-hot encoded atomic number, degree, formal charge, Hs, hybridization, aromaticity, ring membership
- **Edge features**: One-hot bond types + self-loops
- **Network**: 3-layer multi-head GAT (4 heads), 256 hidden dim, global mean pooling
- **Classification head**: 256 → 128 → ReLU → 1 (with BCEWithLogitsLoss)

See `plots/supervised_gnn_architecture.png` for a visual overview.

## Script Index

| Script | Description |
|--------|-------------|
| `01_generate_PKS_products.py` | Generate PKS products using `bcs` library |
| `02_generate_PKS_augmentations.py` | Generate hard-negative triplets |
| `03_create_train_val_test_splits.py` | Connected-component train/val/test splits |
| `04_generate_extender_ood_eval_set.py` | OOD eval set from held-out extender codes |
| `05_generate_mixed_extender_ood_set.py` | Mixed-extender OOD eval set |
| `06_generate_methyltransferase_eval_set.py` | Methyltransferase-modified eval set |
| `07_train_gnn_classifier.py` | Distributed supervised GNN training |
| `08_test_gnn_classifier.py` | Test set evaluation (GNN vs ECFP4) |
| `09_evaluate_ood_recall.py` | OOD recall evaluation |
| `10_run_inference.py` | Single-molecule inference |

## Dependencies

- `torch` / `torch_geometric`: GNN implementation
- `rdkit`: Molecular parsing and graph featurization
- `bcs`: PKS product generation (RetroTide)
- `doranet`: Enzymatic and synthetic reaction network generation
- `mpi4py`: Distributed processing for augmentation generation
- `pandas`: Data handling (parquet format)
- `scikit-learn`: ECFP4 baseline (LogisticRegression, metrics)
- `tqdm`: Progress bars

## Project Structure

```
PolyketideClassifier/
├── scripts/           # Numbered pipeline scripts (01-10)
├── data/
│   ├── raw/           # Original PKS products
│   ├── interim/       # Intermediate files
│   ├── processed/     # Final datasets, augmentation pairs, eval sets
│   ├── evals/         # Curated evaluation sets
│   ├── train/         # Training split (supcon_train.parquet)
│   ├── val/           # Validation split (supcon_val.parquet)
│   └── test/          # Test split (supcon_test.parquet)
├── models/
│   └── supervised_gnn/  # Checkpoints and evaluation results
├── plots/             # Architecture diagrams and training curves
├── notebooks/         # Jupyter notebooks for exploration
├── tests/             # pytest test suite for data integrity
└── LICENSE            # MIT License
```

## License

MIT License. See [LICENSE](LICENSE) for details.
