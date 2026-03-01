#!/usr/bin/env python3
"""
Test Supervised GNN Classifier vs ECFP4 Baseline

Loads a trained SupervisedGNNClassifier checkpoint, runs direct inference on
the test set, and compares against an ECFP4 logistic regression baseline.

Unlike the SupCon pipeline (which requires embedding extraction + linear probe),
the supervised GNN makes predictions directly via sigmoid(logit).

Prerequisites:
    - Run scripts/07_train_gnn_classifier.py to produce:
        models/supervised_gnn/best_model.pt

Usage:
    python scripts/08_test_gnn_classifier.py
    python scripts/08_test_gnn_classifier.py --checkpoint models/supervised_gnn/checkpoint_epoch_020.pt
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, rdchem

RDLogger.DisableLog("rdApp.*")


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Supervised GNN Classifier vs ECFP4 Baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, default="models/supervised_gnn/best_model.pt",
        help="Path to supervised GNN checkpoint"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/",
        help="Path to data directory containing train/test subdirs"
    )
    parser.add_argument(
        "--output_path", type=str, default="models/supervised_gnn/test_comparison.json",
        help="Path for saving comparison JSON"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    return parser.parse_args()


# =============================================================================
# Graph Featurization
# =============================================================================

ATOM_TYPES = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
NUM_HS = [0, 1, 2, 3, 4]
HYBRIDIZATIONS = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]


def _build_mapping(values: Iterable) -> Dict:
    return {value: idx for idx, value in enumerate(values)}


ATOM_MAP = _build_mapping(ATOM_TYPES)
DEGREE_MAP = _build_mapping(DEGREES)
CHARGE_MAP = _build_mapping(FORMAL_CHARGES)
NUM_H_MAP = _build_mapping(NUM_HS)
HYB_MAP = {hyb: idx for idx, hyb in enumerate(HYBRIDIZATIONS)}
BOND_MAP = {bond: idx for idx, bond in enumerate(BOND_TYPES)}

EDGE_FEAT_DIM = len(BOND_TYPES) + 1


def _one_hot(value, mapping: Dict) -> np.ndarray:
    size = len(mapping) + 1
    vec = np.zeros(size, dtype=np.float32)
    vec[mapping.get(value, len(mapping))] = 1.0
    return vec


def atom_to_feature(atom: rdchem.Atom) -> np.ndarray:
    feats = [
        _one_hot(atom.GetAtomicNum(), ATOM_MAP),
        _one_hot(atom.GetTotalDegree(), DEGREE_MAP),
        _one_hot(atom.GetFormalCharge(), CHARGE_MAP),
        _one_hot(atom.GetTotalNumHs(includeNeighbors=True), NUM_H_MAP),
        _one_hot(atom.GetHybridization(), HYB_MAP),
        np.array([atom.GetIsAromatic()], dtype=np.float32),
        np.array([atom.IsInRing()], dtype=np.float32),
    ]
    return np.concatenate(feats, axis=0)


def bond_to_feature(bond) -> np.ndarray:
    vec = np.zeros(EDGE_FEAT_DIM, dtype=np.float32)
    if bond is None:
        vec[-1] = 1.0
    else:
        vec[BOND_MAP.get(bond.GetBondType(), EDGE_FEAT_DIM - 1)] = 1.0
    return vec


def smiles_to_graph(smiles: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    n = mol.GetNumAtoms()
    if n == 0:
        raise ValueError(f"SMILES with no atoms: {smiles}")

    node_feat = np.vstack([atom_to_feature(atom) for atom in mol.GetAtoms()]).astype(np.float32)

    edges: List[Tuple[int, int]] = []
    edge_feat: List[np.ndarray] = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_to_feature(bond)
        edges.append((u, v))
        edges.append((v, u))
        edge_feat.append(feat)
        edge_feat.append(feat)

    loop = bond_to_feature(None)
    for i in range(n):
        edges.append((i, i))
        edge_feat.append(loop)

    edge_index = np.array(edges, dtype=np.int64).T
    edge_attr = np.vstack(edge_feat).astype(np.float32)
    return node_feat, edge_index, edge_attr


def smiles_to_ecfp4(smiles: str, nbits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nbits)
    return np.array(fp, dtype=np.float32)


# =============================================================================
# Dataset and DataLoader
# =============================================================================

@dataclass
class GraphSample:
    node_feat: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    label: int


class MolecularGraphDataset(Dataset):
    def __init__(self, parquet_path: str):
        df = pd.read_parquet(parquet_path)
        self.smiles = df["smiles"].astype(str).tolist()
        self.labels = df["label"].to_numpy().astype(np.int64)

        for smi in self.smiles:
            try:
                nf, _, ea = smiles_to_graph(smi)
                self.node_feat_dim = nf.shape[1]
                self.edge_feat_dim = ea.shape[1]
                break
            except ValueError:
                continue
        else:
            raise RuntimeError("No valid molecules found")

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> GraphSample:
        nf, ei, ea = smiles_to_graph(self.smiles[idx])
        return GraphSample(nf, ei, ea, int(self.labels[idx]))


def collate_graphs(batch: List[GraphSample]) -> Dict[str, torch.Tensor]:
    node_feats = []
    edge_indices = []
    edge_attrs = []
    batch_index = []
    labels = []
    offset = 0

    for graph in batch:
        x = torch.from_numpy(graph.node_feat)
        ei = torch.from_numpy(graph.edge_index) + offset
        ea = torch.from_numpy(graph.edge_attr)
        n = x.size(0)

        node_feats.append(x)
        edge_indices.append(ei)
        edge_attrs.append(ea)
        batch_index.append(torch.full((n,), len(labels), dtype=torch.long))
        labels.append(graph.label)
        offset += n

    return {
        "node_feat": torch.cat(node_feats, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_attr": torch.cat(edge_attrs, dim=0),
        "batch": torch.cat(batch_index, dim=0),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# =============================================================================
# Model Architecture
# =============================================================================

def edge_softmax(dst: torch.Tensor, scores: torch.Tensor, num_nodes: int) -> torch.Tensor:
    heads = scores.size(1)
    out = []
    for h in range(heads):
        s = scores[:, h]
        max_vals = torch.full((num_nodes,), -float("inf"), device=s.device)
        max_vals.scatter_reduce_(0, dst, s, reduce="amax")
        s = s - max_vals[dst]
        exp_s = torch.exp(s)
        denom = torch.zeros(num_nodes, device=s.device).scatter_add_(0, dst, exp_s)
        out.append(exp_s / (denom[dst] + 1e-16))
    return torch.stack(out, dim=1)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int, edge_dim: int, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.lin = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(heads, out_dim))
        self.att_dst = nn.Parameter(torch.Tensor(heads, out_dim))
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        self.edge_proj = nn.Linear(edge_dim, heads, bias=False)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)
        nn.init.xavier_uniform_(self.edge_proj.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        num_nodes = h.size(0)
        h = h.view(num_nodes, self.heads, self.out_dim)

        att_src = (h * self.att_src).sum(dim=-1)
        att_dst = (h * self.att_dst).sum(dim=-1)

        src, dst = edge_index
        edge_logits = att_src[src] + att_dst[dst] + self.edge_proj(edge_attr)
        edge_logits = F.leaky_relu(edge_logits, 0.2)

        edge_alpha = edge_softmax(dst, edge_logits, num_nodes)
        edge_alpha = F.dropout(edge_alpha, p=self.dropout, training=self.training)

        out = h[src] * edge_alpha.unsqueeze(-1)
        agg = torch.zeros(num_nodes, self.heads, self.out_dim, device=x.device)
        agg.index_add_(0, dst, out)

        return agg.mean(dim=1) + self.bias


class SupervisedGNNClassifier(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 256,
        heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(node_dim, hidden_dim)

        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                GraphAttentionLayer(hidden_dim, hidden_dim, heads, edge_dim, dropout)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = dropout
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(node_feat)

        for gat, norm in zip(self.gat_layers, self.layer_norms):
            residual = x
            x = gat(x, edge_index, edge_attr)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual

        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        graph_embed = torch.zeros(num_graphs, self.hidden_dim, device=x.device)
        counts = torch.zeros(num_graphs, device=x.device)
        graph_embed.index_add_(0, batch, x)
        counts.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
        graph_embed = graph_embed / counts.clamp_min(1.0).unsqueeze(-1)

        logit = self.cls_head(graph_embed)
        return logit, graph_embed


# =============================================================================
# Checkpoint Loading
# =============================================================================

def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> SupervisedGNNClassifier:
    """Load SupervisedGNNClassifier from a training checkpoint.

    Infers architecture from saved args or from state_dict keys.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint["model_state_dict"]

    # Prefer saved args if available
    if "args" in checkpoint:
        saved_args = checkpoint["args"]
        node_dim = model_state["input_proj.weight"].shape[1]
        edge_dim = model_state["gat_layers.0.edge_proj.weight"].shape[1]
        model = SupervisedGNNClassifier(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=saved_args["hidden_dim"],
            heads=saved_args["num_heads"],
            num_layers=saved_args["num_layers"],
            dropout=saved_args["dropout"],
        )
    else:
        # Infer from state_dict
        hidden_dim = model_state["input_proj.weight"].shape[0]
        node_dim = model_state["input_proj.weight"].shape[1]
        edge_dim = model_state["gat_layers.0.edge_proj.weight"].shape[1]
        num_layers = sum(1 for k in model_state if k.startswith("gat_layers.") and k.endswith(".lin.weight"))
        heads = model_state["gat_layers.0.att_src"].shape[0]
        model = SupervisedGNNClassifier(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            num_layers=num_layers,
        )

    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("val_loss", float("nan"))
    print(f"  Loaded checkpoint from epoch {epoch} (val loss: {val_loss:.4f})")

    return model


# =============================================================================
# GNN Inference
# =============================================================================

@torch.no_grad()
def run_gnn_inference(
    model: SupervisedGNNClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run direct GNN inference. Returns (probs, labels)."""
    model.eval()
    all_probs = []
    all_labels = []

    for batch in loader:
        node_feat = batch["node_feat"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        batch_idx = batch["batch"].to(device)

        logits, _ = model(node_feat, edge_index, batch_idx, edge_attr)
        probs = torch.sigmoid(logits).squeeze(1)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(batch["labels"].numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


# =============================================================================
# Evaluation
# =============================================================================

def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics from probabilities."""
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "auprc": float(average_precision_score(labels, probs)),
        "auroc": float(roc_auc_score(labels, probs)),
        "f1": float(f1_score(labels, preds)),
    }


def print_comparison_table(
    gnn_metrics: Dict[str, float],
    ecfp4_metrics: Dict[str, float],
) -> None:
    metric_names = ["accuracy", "auprc", "auroc", "f1"]
    labels = ["Accuracy", "AUPRC", "AUROC", "F1"]

    print("\n" + "=" * 72)
    print("Test Set Comparison: Supervised GNN (direct) vs ECFP4 Probe")
    print("=" * 72)
    print(f"{'Metric':<12} {'GNN':>12} {'ECFP4':>12} {'Delta':>12}")
    print("-" * 72)

    for label, key in zip(labels, metric_names):
        gnn_val = gnn_metrics[key]
        ecfp4_val = ecfp4_metrics[key]
        delta = gnn_val - ecfp4_val
        sign = "+" if delta >= 0 else ""
        print(f"{label:<12} {gnn_val:>12.4f} {ecfp4_val:>12.4f} {sign}{delta:>11.4f}")

    print("=" * 72)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Supervised GNN Classifier — Test Set Evaluation")
    print("=" * 72)
    print(f"  Device: {device}")
    print(f"  Checkpoint: {args.checkpoint}")

    # --- Load GNN model ---
    print(f"\nLoading supervised GNN from {args.checkpoint}...")
    model = load_model_from_checkpoint(args.checkpoint, device)

    # --- Load test data ---
    data_dir = Path(args.data_dir)
    test_path = data_dir / "test" / "supcon_test.parquet"
    print(f"\nLoading test data from {test_path}...")
    test_ds = MolecularGraphDataset(str(test_path))
    print(f"  Test samples: {len(test_ds)}, PKS ratio: {test_ds.labels.mean():.3f}")

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=args.num_workers,
    )

    # --- GNN direct inference ---
    print("\nRunning GNN inference on test set...")
    gnn_probs, test_labels = run_gnn_inference(model, test_loader, device)
    gnn_metrics = compute_metrics(test_labels, gnn_probs)
    print(f"  GNN AUPRC: {gnn_metrics['auprc']:.4f}")

    # --- ECFP4 baseline ---
    print("\nTraining ECFP4 baseline...")
    train_path = data_dir / "train" / "supcon_train.parquet"
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    print("  Generating ECFP4 fingerprints...")
    train_fps = np.vstack([smiles_to_ecfp4(smi) for smi in train_df["smiles"].astype(str)])
    test_fps = np.vstack([smiles_to_ecfp4(smi) for smi in test_df["smiles"].astype(str)])
    train_labels = train_df["label"].to_numpy()

    print(f"  Training LogisticRegression on {len(train_fps)} samples...")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    clf.fit(train_fps, train_labels)

    ecfp4_probs = clf.predict_proba(test_fps)[:, 1]
    ecfp4_metrics = compute_metrics(test_labels, ecfp4_probs)
    print(f"  ECFP4 AUPRC: {ecfp4_metrics['auprc']:.4f}")

    # --- Comparison ---
    print_comparison_table(gnn_metrics, ecfp4_metrics)

    # --- Save results ---
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison = {
        "checkpoint": args.checkpoint,
        "supervised_gnn_direct": gnn_metrics,
        "ecfp4_probe": ecfp4_metrics,
        "delta": {
            key: gnn_metrics[key] - ecfp4_metrics[key]
            for key in gnn_metrics
        },
    }

    print(f"\nSaving comparison to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
