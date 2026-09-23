# GROVER GNNExplainer

A class `GROVERExplainer` is designed to provide GNNExplainer functionalities in a GROVER model trained to predict binary solubility classes from molecular graphs.

The explanation optimises an edge mask and a node mask so that the masked model prediction remains close to the original sigmoid probability, while encouraging a sparse and low-entropy mask.

To do so, (1) GROVER architecture is modified to accept an optional edge mask and (2) `GROVERExplainer` and `GROVERExplanation` classes are provided. Here below, both steps are detailed.

## 1. GROVER architecture modification

## 2. GROVERExplainer and GROVERExplanation

`GROVERExplainer` uses the default parameters in GNNExplainer and is initialised with the trained GROVER model, the number of epochs and learning rate for the mask optimisation, the type of node and edge mask, if any, and the loss terms.

### `_initialize_mask`

A method responsible for initialising the learnable masks is available. It accepts the number of nodes, node features, edges and the device.

Different node mask types are available (`N` represents the number of atoms and `F` the number of node features in `f_atoms`):

* `object` (`shape [N, 1]`): the mask contains one importance value for each node (atom), which is applied to all its features. With this mask type, we try to answer which atoms are important for a certain molecule prediction.

* `attributes` (`shape [N, F]`): the mask contains an importance value for each atom-feature pair. We try to answer which features are important for each atom.

* `common_attributes` (`shape [1, F]`): the mask contains one importance value for each feature, shared across all atoms. We try to answer which features are important in the graph as a whole.

There is only one edge mask type, `object` (`shape [E]`), where `E` is the number of real directed GROVER bonds. It contains one value for each directed edge in the GROVER molecular graph.

### `_get_grover_*_mask`

Two methods, `_get_grover_edge_mask` and `_get_grover_node_mask`, are available to prepare the edge and node masks, respectively, when they are enabled.

They apply the sigmoid function to the learnable mask values and add the required padding term. For node masks, the padding value is 1 so that the padding atom remains unchanged. For edge masks, the padding value is 0.

### `_loss`

This method computes the objective used to optimise the masks. It consists of a prediction-preservation loss and two types of regularisation:

1. **Prediction loss:** The mean squared error between the original prediction and the prediction obtained with the current masks. It encourages the masked prediction to remain similar to the original prediction.

2. **Size regularisation:** For the edge mask, it sums all mask values and multiplies the result by `edge_size` (default: 0.005). For the node mask, it averages all mask values and multiplies the result by `node_feat_size` (default: 1.0). This encourages smaller mask values and therefore sparser explanations.

3. **Entropy regularisation:** It computes the binary entropy of a mask value $m$ using

   $$
   H(m)=-m\log(m)-(1-m)\log(1-m).
   $$

   This function reaches its maximum at $m=0.5$ and its minimum at $m=0$ and $m=1$. Since this term is minimised, it encourages mask values close to 0 or 1 rather than values around 0.5. The mean entropy is multiplied by `edge_ent` for the edge mask and `node_feat_ent` for the node mask. The default values are 1.0 and 0.1, respectively.

#### Hard edge and node masks

During the first optimisation epoch, the gradient of each learnable mask element is inspected after backpropagation. `hard_edge_mask` and `hard_node_mask` identify the mask elements that received a non-zero gradient:

* `True`: the mask element received a gradient and therefore has a gradient path to the model output.
* `False`: the mask element did not receive a gradient and therefore does not contribute to the optimisation of the prediction.

These masks follow the procedure used by PyG's GNNExplainer. During post-processing, mask elements with `hard_*_mask = False` are set to zero and are therefore excluded from the final explanation.

### `_train`

This method sets the GROVER model to evaluation mode, initialises the edge and node masks and prepares the optimiser. It calculates the original prediction, which is used as the target, and jointly optimises the available masks in an epoch loop.

When both a node mask and an edge mask are enabled, both masks are included in the same optimiser and are therefore optimised jointly using the same loss.

At the end of the first optimisation epoch, it calculates `hard_edge_mask` and `hard_node_mask` by identifying the mask elements whose gradients are non-zero.

### `_post_process_mask`

It processes the masks by applying the sigmoid function and removing attributions for elements that did not receive gradients during the first optimisation step (`hard_*_mask = False`).

### `forward`

It optimises the masks using the `_train` method, then post-processes the masks and stores them in a `GROVERExplanation` object together with the original GROVER graph and the node mask type. The resulting `GROVERExplanation` is validated and returned.

___

The `GROVERExplanation` class holds the results produced by `GROVERExplainer`.

* It is initialised with the input GROVER graph, the optimised edge mask and node mask, and the `node_mask_type`.

* The method `validate` validates the dimensions of the masks according to their mask type.

* The method `get_edge_index` converts the GROVER-style directed bond representation into a PyG-style `edge_index` matrix. GROVER uses 1-based atom indices and a padding atom/bond at index 0, which are removed and converted to 0-based indices.

* The method `get_undirected_edge_mask` combines the two directed edges corresponding to each chemical bond into one undirected edge attribution. The combination (`reduction`) can be performed using `mean` or `max`.

* The method `get_explanation_subgraph` returns the nodes and edges with non-zero attribution. For `object` and `attributes` node masks, nodes are selected based on their aggregated node attribution. For `common_attributes`, no node-specific selection is performed because the mask is shared across all nodes.

* The method `get_complement_subgraph` returns the nodes and edges with zero attribution. It follows the same logic as `get_explanation_subgraph`, but selects elements with zero attribution instead.

* The method `_get_subgraph` performs the actual extraction of the selected nodes and edges and returns their indices together with the corresponding `edge_index`.

* The method `visualize_feature_importance` visualises feature-level node attributions as a bar plot. It is available for `attributes` and `common_attributes` masks, but not for `object` masks. For an `attributes` mask, the attribution of each feature is obtained by summing its attribution across atoms.

* The method `get_node_importance` returns one aggregated node importance value per atom. For an `object` mask, this corresponds directly to the atom mask. For an `attributes` mask, feature attributions are summed for each atom. `common_attributes` masks cannot provide node-specific importance because the same feature mask is shared across all atoms.

* The method `get_edge_importance` returns the directed edge attribution mask. The values correspond to the individual real directed GROVER bonds.

* The method `get_available_results` returns the main results stored in the explanation, including the node mask, edge mask, node mask type and PyG-style `edge_index`.











