import os
from argparse import Namespace
import os 
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
os.chdir("/home/mabarr/TCruzi_pipeline/external/grover/")
from torch.utils.data import DataLoader
from grover.data import MolCollator
from grover.data import MoleculeDataset
from grover.model.models import GroverFinetuneTask
from grover.util.utils import get_data_from_smiles, load_args, load_scalars, get_model_args
os.chdir("/home/mabarr/TCruzi_pipeline/")
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

args = Namespace(
    data_path="external/grover/solubility_data/solubility_data_categorized.csv",
    save_dir="model_nofeatures_category",
    checkpoint_dir = "model_nofeatures_category",
    checkpoint_paths = ["model_nofeatures_category/fold_0/model_0/model.pt"],
    dataset_type="classification",
    split_type="scaffold_balanced",
    split_sizes=[0.7, 0.15, 0.15],
    metric="accuracy",
    ensemble_size=1,
    num_folds=1,
    ffn_hidden_size=200,
    batch_size=1,
    epochs=60,
    init_lr=0.00015,
    self_attention=True,
    save_smiles_splits=True,
    use_wandb=True,
    wandb_entity="mariabar",
    num_workers=50,
    dropout=0.1,
    ffn_num_layers=1,
    features_path = None,
    max_data_size = None,
    use_compound_names = None,
    seed = 0,
    separate_test_path = None,
    separate_val_path = None,
    folds_file = None,
    val_fold_index = None,
    test_fold_index = None,
    features_scaling=False,
    bond_drop_rate = 0,
    no_cache = True,
    cuda = True,
    features_only=False,
)

debug = info = print
logger = None

def model_data_preparation(current_args, checkpoint_path, smiles):

    test_data = get_data_from_smiles(smiles=smiles, 
                                     skip_invalid_smiles=False, 
                                     args = current_args)

    scaler, features_scaler = load_scalars(checkpoint_path)
    train_args = load_args(checkpoint_path)

    for key, value in vars(train_args).items():
        if not hasattr(current_args, key):
            setattr(current_args, key, value)

    test_data = MoleculeDataset(test_data)

    ## Dataloader
    mol_collator = MolCollator(shared_dict={}, args=current_args)
    data_loader = DataLoader(test_data,
                             batch_size=current_args.batch_size,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=mol_collator)

    state = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False)

    args, loaded_state_dict = state['args'], state['state_dict']

    model_related_args = get_model_args()

    if current_args is not None:
        for key, value in vars(args).items():
            if key in model_related_args:
                setattr(current_args, key, value)

    model = GroverFinetuneTask(current_args)

    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():
        new_param_name = param_name
        if new_param_name not in model_state_dict:
            debug(f'Pretrained parameter "{param_name}" cannot be found in model parameters.')
        elif model_state_dict[new_param_name].shape != loaded_state_dict[param_name].shape:
            debug(f'Pretrained parameter "{param_name}" '
                    f'of shape {loaded_state_dict[param_name].shape} does not match corresponding '
                    f'model parameter of shape {model_state_dict[new_param_name].shape}.')
        else:
            debug(f'Loading pretrained parameter "{param_name}".')
            pretrained_state_dict[new_param_name] = loaded_state_dict[param_name]
    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    return model, data_loader


def grover_to_pyg(b2a, b2revb):

    edge_index = torch.stack([
        b2a,
        b2a[b2revb]
    ])

    return edge_index


class GroverAdapter(nn.Module):
    def __init__(self, grover_model):
        super().__init__()
        self.grover = grover_model

    def pyg_to_grover(self, x, edge_index, edge_attr):

        f_atoms = x
        f_bonds = edge_attr

        # Source atom of each directed bond
        b2a = edge_index[0]

        b2revb = torch.empty(
            edge_index.shape[1],
            dtype=torch.long,
            device=edge_index.device
        )

        for i in range(edge_index.shape[1]):
            source = edge_index[0, i]
            target = edge_index[1, i]

            reverse = torch.where(
                (edge_index[0] == target) &
                (edge_index[1] == source)
            )[0]

            b2revb[i] = reverse[0]

        num_atoms = x.shape[0]

        a2b_list = []

        for atom in range(num_atoms):
            incoming = torch.where(edge_index[1] == atom)[0]
            a2b_list.append(incoming)

        max_bonds = max(len(row) for row in a2b_list)

        a2b = torch.stack([
            torch.cat([
                row,
                torch.zeros(
                    max_bonds - len(row),
                    dtype=torch.long,
                    device=edge_index.device
                )
            ])
            for row in a2b_list
        ])

        a2a = torch.zeros_like(a2b)

        for atom in range(num_atoms):
            bonds = a2b[atom]
            a2a[atom] = edge_index[0, bonds]

        a_scope = [(1, num_atoms-1)]
        b_scope = [(1, edge_index.shape[1] - 1)]

        return (
            f_atoms,
            f_bonds,
            a2b,
            b2a,
            b2revb,
            a_scope,
            b_scope,
            a2a,
        )

    def forward(self, x, edge_index, edge_attr):

        batch = self.pyg_to_grover(
            x,
            edge_index,
            edge_attr
        )

        output = self.grover(
            batch,
            [None]
        )

        return output


class GROVERExplanation:
    """
    Holds the results of a GROVERExplainer explanation.

    Parameters
    ----------
    graph : tuple
        GROVER graph tuple:
        (
            f_atoms,
            f_bonds,
            a2b,
            b2a,
            b2revb,
            a_scope,
            b_scope,
            a2a
        )

    node_mask : torch.Tensor, optional
        Node attribution mask. Possible shapes:
            [num_nodes, 1]
            [num_nodes, num_features]
            [1, num_features]

    edge_mask : torch.Tensor, optional
        Edge attribution mask for real directed GROVER bonds only.
        Shape:
            [num_real_directed_edges]

    node_mask_type : str, optional
        Type of node mask:
            "object"
            "attributes"
            "common_attributes"

    """

    def __init__(
        self,
        graph,
        node_mask=None,
        edge_mask=None,
        node_mask_type=None,
    ):
        self.graph = graph

        self.node_mask = node_mask
        self.edge_mask = edge_mask

        self.node_mask_type = node_mask_type

        (
            self.f_atoms,
            self.f_bonds,
            self.a2b,
            self.b2a,
            self.b2revb,
            self.a_scope,
            self.b_scope,
            self.a2a,
        ) = graph

        # Number of real atoms/bonds, excluding GROVER padding.
        self.num_nodes = self.f_atoms.size(0) - 1
        self.num_edges = self.f_bonds.size(0) - 1

    @property
    def available_explanations(self):
        """Return the explanations currently available."""
        explanations = []

        if self.node_mask is not None:
            explanations.append("node_mask")

        if self.edge_mask is not None:
            explanations.append("edge_mask")

        return explanations

    def validate(self):
        """
        Validate the dimensions of the explanation masks.
        """

        if self.node_mask is not None:

            if self.node_mask_type == "object":
                expected_shape = (self.num_nodes, 1)

                if tuple(self.node_mask.shape) != expected_shape:
                    raise ValueError(
                        f"Object node mask should have shape "
                        f"{expected_shape}, got {self.node_mask.shape}"
                    )

            elif self.node_mask_type == "attributes":
                if self.node_mask.dim() != 2:
                    raise ValueError(
                        "Attribute node mask must have shape "
                        "[num_nodes, num_features]."
                    )

                if self.node_mask.size(0) != self.num_nodes:
                    raise ValueError(
                        f"Attribute node mask should contain "
                        f"{self.num_nodes} nodes, got "
                        f"{self.node_mask.size(0)}."
                    )

            elif self.node_mask_type == "common_attributes":
                if self.node_mask.dim() != 2 or self.node_mask.size(0) != 1:
                    raise ValueError(
                        "Common attribute node mask must have shape "
                        "[1, num_features]."
                    )

            else:
                raise ValueError(
                    f"Unknown node_mask_type: {self.node_mask_type}"
                )

        if self.edge_mask is not None:

            if self.edge_mask.dim() != 1:
                raise ValueError(
                    "Edge mask must have shape [num_edges]."
                )

            if self.edge_mask.size(0) != self.num_edges:
                raise ValueError(
                    f"Edge mask should contain {self.num_edges} "
                    f"real directed edges, got "
                    f"{self.edge_mask.size(0)}."
                )

        return True

    def get_edge_index(self):
        """
        Convert the GROVER directed-bond representation to a
        PyG-style edge_index.

        Returns
        -------
        edge_index : torch.Tensor
            Shape [2, num_real_directed_edges].

        Notes
        -----
        GROVER atom indices are 1-based and contain padding atom 0.
        The returned edge_index is 0-based and excludes padding.
        """

        edge_index = torch.stack(
            [
                self.b2a,
                self.b2a[self.b2revb],
            ],
            dim=0,
        )

        # Remove padding bond.
        edge_index = edge_index[:, 1:]

        # Convert GROVER's 1-based atom indices to 0-based.
        edge_index = edge_index - 1

        return edge_index

    def get_undirected_edge_mask(self, reduction="mean"):
        """
        Combine the two directed GROVER edges corresponding to
        each chemical bond.

        Parameters
        ----------
        reduction : str
            How to combine the two directions:
                "mean"
                "max"

        Returns
        -------
        torch.Tensor
            One attribution value per chemical bond.
        """

        if self.edge_mask is None:
            raise ValueError(
                "The attribute 'edge_mask' is not available."
            )

        # GROVER's real directed edges are:
        # 1 <-> 2
        # 3 <-> 4
        # ...
        #
        # The returned mask therefore contains one value per
        # undirected chemical bond.

        mask = self.edge_mask

        if mask.size(0) % 2 != 0:
            raise ValueError(
                "Expected an even number of real directed edges."
            )

        mask = mask.view(-1, 2)

        if reduction == "mean":
            return mask.mean(dim=1)

        elif reduction == "max":
            return mask.max(dim=1).values

        else:
            raise ValueError(
                f"Unknown reduction: {reduction}"
            )

    def get_explanation_subgraph(self):
        """
        Return the nodes and edges with non-zero attribution.

        This follows the same basic logic as PyG's
        get_explanation_subgraph().
        """

        node_mask = None
        edge_mask = None

        if self.node_mask is not None:
            if self.node_mask_type == "common_attributes":
                # Common attribute masks do not identify individual nodes.
                node_mask = None
            else:
                node_mask = self.node_mask.sum(dim=-1) > 0

        if self.edge_mask is not None:
            edge_mask = self.edge_mask > 0

        return self._get_subgraph(
            node_mask=node_mask,
            edge_mask=edge_mask,
        )

    def get_complement_subgraph(self):
        """
        Return the nodes and edges with zero attribution.
        """

        node_mask = None
        edge_mask = None

        if self.node_mask is not None:
            if self.node_mask_type == "common_attributes":
                node_mask = None
            else:
                node_mask = self.node_mask.sum(dim=-1) == 0

        if self.edge_mask is not None:
            edge_mask = self.edge_mask == 0

        return self._get_subgraph(
            node_mask=node_mask,
            edge_mask=edge_mask,
        )

    def _get_subgraph(self, node_mask=None, edge_mask=None):
        """
        Extract a subgraph from the GROVER graph.

        Returns a dictionary containing the selected atoms and bonds.
        """

        selected_nodes = None
        selected_edges = None

        if node_mask is not None:
            selected_nodes = torch.where(node_mask)[0]

        if edge_mask is not None:
            selected_edges = torch.where(edge_mask)[0]

        return {
            "node_indices": selected_nodes,
            "edge_indices": selected_edges,
            "edge_index": (
                self.get_edge_index()[:, edge_mask]
                if edge_mask is not None
                else None
            ),
        }

    def visualize_feature_importance(
        self,
        feat_labels=None,
        top_k=None,
        path=None,
    ):
        """
        Plot node feature importance.

        Only applicable to feature-level node masks:
            "attributes"
            "common_attributes"
        """

        if self.node_mask is None:
            raise ValueError(
                "The attribute 'node_mask' is not available."
            )

        if self.node_mask_type == "object":
            raise ValueError(
                "Cannot compute feature importance for "
                "an object-level node mask."
            )

        # For attributes:
        # [num_nodes, num_features]
        #
        # For common_attributes:
        # [1, num_features]
        score = self.node_mask.sum(dim=0)

        if feat_labels is None:
            feat_labels = [
                str(i) for i in range(score.numel())
            ]

        if len(feat_labels) != score.numel():
            raise ValueError(
                "Number of feature labels does not match "
                "the number of features."
            )

        if top_k is not None:
            top_k = min(top_k, score.numel())

            values, indices = torch.topk(
                score,
                top_k,
            )

            score = values
            feat_labels = [
                feat_labels[i]
                for i in indices.cpu().tolist()
            ]

        score = score.detach().cpu().numpy()

        plt.figure(figsize=(10, 6))

        plt.bar(
            range(len(score)),
            score,
        )

        plt.xticks(
            range(len(score)),
            feat_labels,
            rotation=90,
        )

        plt.ylabel("Feature importance")
        plt.tight_layout()

        if path is not None:
            plt.savefig(
                path,
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

        else:
            plt.show()

    def get_node_importance(self):
        """
        Return one node importance value per atom.

        For:
            object        -> directly returns the mask
            attributes    -> sums feature attributions
            common_attributes -> not node-specific
        """

        if self.node_mask is None:
            raise ValueError(
                "The attribute 'node_mask' is not available."
            )

        if self.node_mask_type == "common_attributes":
            raise ValueError(
                "Common attribute masks do not provide "
                "node-specific importance."
            )

        return self.node_mask.sum(dim=-1)

    def get_edge_importance(self):
        """
        Return the directed-edge importance mask.
        """

        if self.edge_mask is None:
            raise ValueError(
                "The attribute 'edge_mask' is not available."
            )

        return self.edge_mask

    def get_available_results(self):
        """
        Return a summary of the available explanation results.
        """

        results = {
            "node_mask": self.node_mask,
            "edge_mask": self.edge_mask,
            "node_mask_type": self.node_mask_type,
            "edge_index": (
                self.get_edge_index()
                if self.edge_mask is not None
                else None
            ),
        }

        return results



class GROVERExplainer:
    """
    GNNExplainer-style edge and node explanation for a GROVER binary
    classification model.

    The explanation optimises an edge mask and a node mask so that the masked
    model prediction remains close to the original sigmoid
    probability, while encouraging a sparse and low-entropy mask.

    It uses the deafult parameters in GNNExplainer:

    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
        'EPS': 1e-15,
    }

    Assumes:
        - batch size = 1
        - graph contains one molecule
        - GROVER bond index 0 is padding
        - model returns a single sigmoid probability in eval mode
    """

    def __init__(
        self,
        model,
        epochs=100,
        lr=0.01,
        node_mask_type="attributes",
        edge_mask_type="object",
        edge_size=0.005,
        edge_ent=1.0,
        node_feat_size=1.0,
        node_feat_ent = 0.1,
        eps=1e-15,
    ):
        self.model = model
        self.epochs = epochs
        self.lr = lr

        self.node_mask_type = node_mask_type
        self.edge_mask_type = edge_mask_type

        self.edge_size = edge_size
        self.edge_ent = edge_ent

        self.node_feat_size = node_feat_size
        self.node_feat_ent = node_feat_ent

        self.eps = eps

        self.node_mask = None
        self.edge_mask = None

        self.hard_node_mask = None
        self.hard_edge_mask = None

    def _initialize_mask(
            self, 
            num_nodes, 
            num_node_features,
            num_edges, 
            device
    ):
        """
        Initialise the learnable node and edge masks following PyG GNNExplainer.

        num_edges here is the number of real directed edges, excluding
        GROVER's padding bond at index 0.
        num_nodes here is the number of nodes, excluding
        GROVER's padding bond at index 0.
        num_node_features here is the number of node features
        """

        # Node mask
        std = 0.1
        if self.node_mask_type is None:
            self.node_mask = None

        elif self.node_mask_type == "object":
            self.node_mask = nn.Parameter(
                torch.randn(num_nodes, 1, device=device) * std
            )

        elif self.node_mask_type == "attributes":
            self.node_mask = nn.Parameter(
                torch.randn(
                    num_nodes,
                    num_node_features,
                    device=device
                ) * std
            )

        elif self.node_mask_type == "common_attributes":
            self.node_mask = nn.Parameter(
                torch.randn(
                    1,
                    num_node_features,
                    device=device
                ) * std
            )

        else:
            raise ValueError(
                f"Unknown node_mask_type: {self.node_mask_type}"
            )

        # Edge mask
        if self.edge_mask_type is None:
            self.edge_mask = None

        elif self.edge_mask_type == "object":
            std = math.sqrt(2.0 / num_nodes)

            self.edge_mask = nn.Parameter(
                torch.randn(
                    num_edges,
                    device=device
                ) * std
            )

        else:
            raise ValueError(
                f"Unknown edge_mask_type: {self.edge_mask_type}"
            )

        self.hard_node_mask = None
        self.hard_edge_mask = None

    def _get_grover_edge_mask(self):
        """
        Add GROVER's padding entry at index 0.

        self.edge_mask contains only real directed edges.
        GROVER expects:
            index 0 -> padding
            index 1..E -> real directed edges
        """

        if self.edge_mask is None:
            return None

        edge_mask = self.edge_mask.sigmoid()

        padding = torch.zeros(
            1,
            device=self.edge_mask.device,
            dtype=self.edge_mask.dtype,
        )

        return torch.cat([padding, edge_mask], dim=0)

    def _get_grover_node_mask(self):
        """
        Add GROVER's padding entry at index 0.

        self.node_mask contains only real nodes.
        GROVER expects:
            index 0 -> padding
            index 1..N -> real nodes
        """

        if self.node_mask is None:
            return None

        node_mask = self.node_mask.sigmoid()

        if self.node_mask_type == "common_attributes":
            return node_mask

        # padding is 1 here because we want f_atoms[0] to remain unchanged
        padding = torch.ones(
            1,
            node_mask.size(1),
            device=node_mask.device,
            dtype=node_mask.dtype,
        )

        return torch.cat([padding, node_mask], dim=0)

    def _prediction_loss(self, prediction, target):
        """
        Match the masked prediction to the original sigmoid probability.
        """

        prediction = prediction.view_as(target)

        return F.mse_loss(prediction, target)

    def _loss(self, prediction, target):
        """
        Prediction-preservation loss + GNNExplainer-style
        edge size and entropy regularisation.
        """

        # Keep the prediction close to the original probability.
        loss = self._prediction_loss(prediction, target)

        # PyG only starts applying these terms after determining
        # which nodes and edges have non-zero gradients in the first epoch.
        if self.hard_edge_mask is not None:
            assert self.edge_mask is not None
            m = self.edge_mask[self.hard_edge_mask].sigmoid()
            # Edge-size regularisation.
            loss = loss + self.edge_size * m.sum()

            # Edge entropy regularisation.
            ent = (
                -m * torch.log(m + self.eps)
                - (1.0 - m) * torch.log(1.0 - m + self.eps)
            )
            loss = loss + self.edge_ent * ent.mean()

        if self.hard_node_mask is not None:
            assert self.node_mask is not None
            m = self.node_mask[self.hard_node_mask].sigmoid()
            loss = loss + self.node_feat_size * m.mean()
            ent = (
                -m * torch.log(m + self.eps)
                - (1.0 - m) * torch.log(1.0 - m + self.eps)
                )
            loss = loss + self.node_feat_ent * ent.mean()

        return loss

    def _train(self, graph, features_batch):
        """
        Optimise the node and edge masks for one GROVER graph.

        Parameters
        ----------
        graph : tuple
            GROVER graph tuple:
            (
                f_atoms,
                f_bonds,
                a2b,
                b2a,
                b2revb,
                a_scope,
                b_scope,
                a2a
            )
        """

        self.model.eval()

        (
            f_atoms,
            f_bonds,
            a2b,
            b2a,
            b2revb,
            a_scope,
            b_scope,
            a2a,
        ) = graph

        device = f_atoms.device

        # GROVER contains one padding atom and one padding bond.
        num_nodes = f_atoms.size(0) - 1
        num_features = f_atoms.size(1)
        num_edges = f_bonds.size(0) - 1

        self._initialize_mask(
            num_nodes=num_nodes,
            num_node_features=num_features,
            num_edges=num_edges,
            device=device,
        )

        parameters = []
        if self.node_mask is not None:
            parameters.append(self.node_mask)
        if self.edge_mask is not None:
            parameters.append(self.edge_mask)

        optimizer = torch.optim.Adam(
            parameters,
            lr=self.lr,
        )

        # Define the target - original prediction
        with torch.no_grad():
            original_prediction = self.model(graph, features_batch)

        target = original_prediction.detach()

        # Training
        for epoch in tqdm(range(self.epochs)):

            optimizer.zero_grad()

            # Convert learnable logits to probabilities and add
            # GROVER's padding entry.
            grover_edge_mask = self._get_grover_edge_mask()
            grover_node_mask = self._get_grover_node_mask()

            if grover_node_mask is not None:
                masked_f_atoms = f_atoms * grover_node_mask
            else:
                masked_f_atoms = f_atoms

            masked_graph = (
                masked_f_atoms,
                f_bonds,
                a2b,
                b2a,
                b2revb,
                a_scope,
                b_scope,
                a2a,
            )

            # Forward pass with masked messages.
            prediction = self.model(
                masked_graph,
                features_batch,
                edge_mask=grover_edge_mask,
            )

            # Prediction preservation + regularisation.
            loss = self._loss(
                prediction,
                target,
            )

            loss.backward()
            optimizer.step()

            # Match GNNExplainer's hard-edge-mask procedure:
            # determine which edges actually received gradients
            # during the first optimisation step.
            if epoch == 0 and self.edge_mask is not None:
                if self.edge_mask.grad is None:
                    raise ValueError(
                        "Edge mask did not receive gradients. "
                        "Check that edge_mask is propagated through "
                        "all GROVER message-passing operations."
                    )

                self.hard_edge_mask = (
                    self.edge_mask.grad != 0.0
                )

            if epoch == 0 and self.node_mask is not None:
                if self.node_mask.grad is None:
                    raise ValueError(
                        "Node mask did not receive gradients. "
                        "Check that node features are used by GROVER."
                    )

                self.hard_node_mask = (
                    self.node_mask.grad != 0.0
                )

    def _post_process_mask(
        self,
        mask,
        hard_mask=None
    ):
        """
        Post-process a mask by removing attributions for
        elements that did not receive gradients during the
        first optimisation step.
        """
        if mask is None:
            return mask

        mask = mask.detach()
        
        mask = mask.sigmoid()

        # At the beginning of optimisation, if an edge has no gradient path to the prediction, 
        # it is not part of the relevant computation and can be excluded from the explanation. 
        if hard_mask is not None and mask.size(0) == hard_mask.size(0):
            mask[~hard_mask] = 0.0

        return mask

    def forward(self, graph, features_batch):

        self._train(graph, features_batch)

        node_mask = self._post_process_mask(
            self.node_mask,
            self.hard_node_mask
        )

        edge_mask = self._post_process_mask(
            self.edge_mask,
            self.hard_edge_mask
        )

        explanation = GROVERExplanation(
            graph=graph,
            node_mask=node_mask,
            edge_mask=edge_mask,
            node_mask_type=self.node_mask_type,
        )

        explanation.validate()

        return explanation
