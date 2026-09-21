import os
import random
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Makes results more reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Molecular graph construction
# ============================================================

def atom_features(atom):
    """
    Atom-level features.

    The resulting vector contains:
        atomic number
        degree
        formal charge
        number of hydrogens
        aromaticity
        hybridisation
        ring membership
    """

    atomic_number = atom.GetAtomicNum()
    degree = atom.GetDegree()
    formal_charge = atom.GetFormalCharge()
    num_h = atom.GetTotalNumHs()

    aromatic = int(atom.GetIsAromatic())
    in_ring = int(atom.IsInRing())

    hybridisation = atom.GetHybridization()
    hybridisation_map = {
        Chem.HybridizationType.SP: 0,
        Chem.HybridizationType.SP2: 1,
        Chem.HybridizationType.SP3: 2,
        Chem.HybridizationType.SP3D: 3,
        Chem.HybridizationType.SP3D2: 4,
    }

    hybridisation_value = hybridisation_map.get(hybridisation, 5)

    return [
        atomic_number,
        degree,
        formal_charge,
        num_h,
        aromatic,
        hybridisation_value,
        in_ring,
    ]


def bond_features(bond):
    """
    Bond-level features.

    GCNConv itself does not use edge attributes, so these are
    not currently used by the model. They are included here
    in case you later want to switch to a model that uses them.
    """

    bond_type = bond.GetBondType()

    bond_type_map = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3,
    }

    return [
        bond_type_map.get(bond_type, 4),
        int(bond.GetIsAromatic()),
        int(bond.IsInRing()),
    ]


def smiles_to_graph(smiles, label):
    """
    Convert a SMILES string into a PyTorch Geometric Data object.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    # ----------------------------
    # Node features
    # ----------------------------

    x = torch.tensor(
        [atom_features(atom) for atom in mol.GetAtoms()],
        dtype=torch.float
    )

    # ----------------------------
    # Edges
    # ----------------------------

    edges = []
    edge_features = []

    for bond in mol.GetBonds():

        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Molecular graphs are represented as undirected graphs,
        # therefore add both directions.
        edges.append([i, j])
        edges.append([j, i])

        bf = bond_features(bond)
        edge_features.append(bf)
        edge_features.append(bf)

    if len(edges) > 0:
        edge_index = torch.tensor(
            edges,
            dtype=torch.long
        ).t().contiguous()

        edge_attr = torch.tensor(
            edge_features,
            dtype=torch.float
        )
    else:
        # Molecules with a single atom
        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long
        )

        edge_attr = torch.empty(
            (0, 3),
            dtype=torch.float
        )

    y = torch.tensor([label], dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y
    )


# ============================================================
# Scaffold-balanced split
# ============================================================

def get_scaffold(smiles):
    """
    Calculate the Bemis-Murcko scaffold for a molecule.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=False
    )

    return scaffold


def scaffold_balanced_split(
    df,
    split_sizes=(0.8, 0.1, 0.1),
    seed=62
):
    """
    Split molecules into train/validation/test using molecular
    scaffolds.

    All molecules sharing the same scaffold remain in the same
    split, preventing scaffold leakage.

    The scaffolds are sorted by group size and assigned to the
    partition that is currently furthest below its target size.
    """

    assert abs(sum(split_sizes) - 1.0) < 1e-6

    rng = random.Random(seed)

    df = df.copy()

    scaffolds = {}

    for idx, smiles in enumerate(df["SMILES"]):

        scaffold = get_scaffold(smiles)

        if scaffold is None:
            scaffold = f"INVALID_{idx}"

        if scaffold not in scaffolds:
            scaffolds[scaffold] = []

        scaffolds[scaffold].append(idx)

    # Shuffle first so equally sized scaffolds are randomised
    scaffold_items = list(scaffolds.items())
    rng.shuffle(scaffold_items)

    # Larger scaffolds first
    scaffold_items.sort(
        key=lambda x: len(x[1]),
        reverse=True
    )

    target_sizes = np.array(split_sizes) * len(df)

    split_indices = {
        "train": [],
        "val": [],
        "test": []
    }

    current_sizes = np.zeros(3)

    for scaffold, indices in scaffold_items:

        group_size = len(indices)

        # Determine which split is furthest below its target
        deficits = target_sizes - current_sizes

        # Prefer splits that can accommodate the entire scaffold
        possible = np.where(
            current_sizes + group_size <= target_sizes
        )[0]

        if len(possible) > 0:
            split_idx = possible[np.argmax(deficits[possible])]
        else:
            split_idx = np.argmax(deficits)

        split_name = ["train", "val", "test"][split_idx]

        split_indices[split_name].extend(indices)

        current_sizes[split_idx] += group_size

    # Shuffle molecules inside each split
    for split_name in split_indices:
        rng.shuffle(split_indices[split_name])

    print("\nScaffold-balanced split:")
    print(f"Total molecules: {len(df)}")

    for split_name in ["train", "val", "test"]:
        n = len(split_indices[split_name])
        print(
            f"{split_name:5s}: {n:6d} "
            f"({100 * n / len(df):5.2f}%)"
        )

    return split_indices


# ============================================================
# GCN model
# ============================================================

class GCN(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=3,
        dropout=0.2
    ):
        super().__init__()

        self.convs = nn.ModuleList()

        # First GCN layer
        self.convs.append(
            GCNConv(input_dim, hidden_dim)
        )

        # Additional GCN layers
        for _ in range(num_layers - 1):
            self.convs.append(
                GCNConv(hidden_dim, hidden_dim)
            )

        self.dropout = dropout

        # Graph-level classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x, edge_index, batch):

        for conv in self.convs:

            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(
                x,
                p=self.dropout,
                training=self.training
            )

        # Convert node representations to a graph representation
        x = global_mean_pool(x, batch)

        # Binary classification logits
        out = self.classifier(x)

        return out


# ============================================================
# Training / evaluation
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        logits = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        loss = criterion(logits, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

        predictions = logits.argmax(dim=1)

        total_correct += (
            predictions == batch.y
        ).sum().item()

        total_examples += batch.num_graphs

    loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in loader:

        batch = batch.to(device)

        logits = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        loss = criterion(logits, batch.y)

        total_loss += loss.item() * batch.num_graphs

        predictions = logits.argmax(dim=1)

        total_correct += (
            predictions == batch.y
        ).sum().item()

        total_examples += batch.num_graphs

    loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return loss, accuracy


# ============================================================
# Main
# ============================================================

def main(args):

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    df = pd.read_csv(args.csv)

    required_columns = ["SMILES", "logS_category"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"CSV must contain column '{column}'"
            )

    df = df[required_columns].copy()

    # Remove missing values
    df = df.dropna(
        subset=["SMILES", "logS_category"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Make sure labels are binary
    # --------------------------------------------------------

    unique_labels = sorted(
        df["logS_category"].unique()
    )

    print("\nLabels found:")
    print(unique_labels)

    if len(unique_labels) != 2:
        raise ValueError(
            "logS_category must contain exactly two classes."
        )

    # Convert arbitrary binary labels to 0/1
    label_mapping = {
        unique_labels[0]: 0,
        unique_labels[1]: 1
    }

    df["label"] = df["logS_category"].map(
        label_mapping
    )

    print("\nLabel mapping:")
    print(label_mapping)

    # --------------------------------------------------------
    # Scaffold split
    # --------------------------------------------------------

    split_indices = scaffold_balanced_split(
        df,
        split_sizes=(0.8, 0.1, 0.1),
        seed=args.seed
    )

    train_df = df.iloc[
        split_indices["train"]
    ].copy()

    val_df = df.iloc[
        split_indices["val"]
    ].copy()

    test_df = df.iloc[
        split_indices["test"]
    ].copy()

    # Save split assignment
    split_assignment = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test")
        ]
    )

    split_assignment.to_csv(
        os.path.join(
            args.output_dir,
            "split_assignments.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Convert SMILES to graphs
    # --------------------------------------------------------

    print("\nConverting SMILES to graphs...")

    def make_dataset(dataframe):

        graphs = []

        for _, row in dataframe.iterrows():

            graph = smiles_to_graph(
                row["SMILES"],
                int(row["label"])
            )

            graphs.append(graph)

        return graphs

    train_dataset = make_dataset(train_df)
    val_dataset = make_dataset(val_df)
    test_dataset = make_dataset(test_df)

    print(
        f"Train graphs: {len(train_dataset)}"
    )
    print(
        f"Validation graphs: {len(val_dataset)}"
    )
    print(
        f"Test graphs: {len(test_dataset)}"
    )

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    input_dim = train_dataset[0].x.shape[1]

    model = GCN(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    print("\nModel:")
    print(model)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    # Calculate class weights from training data only
    train_labels = train_df["label"].values

    class_counts = np.bincount(
        train_labels,
        minlength=2
    )

    class_weights = len(train_labels) / (
        2.0 * class_counts
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float
    ).to(device)

    print("\nTraining class counts:")
    print(class_counts)

    print("\nClass weights:")
    print(class_weights.cpu().numpy())

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = []

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        print(
            f"Epoch {epoch:03d} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_acc:.4f}"
        )

        # Save best model based on validation loss
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                os.path.join(
                    args.output_dir,
                    "best_model.pt"
                )
            )

    # --------------------------------------------------------
    # Save learning curves
    # --------------------------------------------------------

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        os.path.join(
            args.output_dir,
            "learning_curves.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    model.load_state_dict(
        torch.load(
            os.path.join(
                args.output_dir,
                "best_model.pt"
            ),
            map_location=device
        )
    )

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print("\n==============================")
    print("Best epoch:", best_epoch)
    print("Best validation loss:", best_val_loss)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)
    print("==============================")

    test_results = pd.DataFrame([{
        "best_epoch": best_epoch,
        "validation_loss": best_val_loss,
        "test_loss": test_loss,
        "test_accuracy": test_acc
    }])

    test_results.to_csv(
        os.path.join(
            args.output_dir,
            "test_results.csv"
        ),
        index=False
    )

    print(
        f"\nResults saved to: {args.output_dir}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Input CSV containing SMILES and logS_category"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="gcn_solubility_results"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=3
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    main(args)