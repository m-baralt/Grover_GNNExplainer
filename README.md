# GROVER GNNExplainer

A class GROVERExplainer is designed to provide GNNExplainer functionalities in a GROVER model trained to predict binary solubility classes from molecular graphs.

The explanation optimises an edge mask and a node mask so that the masked model prediction remains close to the original sigmoid probability, while encouraging a sparse and low-entropy mask.

To do so, (1) GROVER architecture is modified to accept an optional edge mask and (2) GROVERExplainer and GROVERExplanation classes are provided. Here below, both steps are detailed. 

## 1. GROVER architecture modification

## 2. GROVERExplainer and GROVERExplanation

GROVERExplainer uses the default parameters in GNNExplainer and it is initialized with the trained GROVER model, the number of epochs and learning rate for the masks model, the type of node and edge mask if any, and the loss terms. 

### `_initialize_mask`

A method responsible for initialising the learnable masks is available. It accepts the number of nodes, node features, edges and the device. 

Different node mask types are available (`N`represents the number of atoms and `F` the number of node features in `f_atoms`): 

* `object` (`shape [N, 1]`): the mask contains an importance value for each node (atom) and its features. With this mask type, we try to answer which atoms are important for a certain molecule prediction. 

* `attributes` (`shape [N, F]`): the mask contains an importance value for each atom feature-pair. We try to answer which features are important for each atom. 

* `common_attributes`: (`shape [1, F]`): the mask contains a value for each feature, common in all atoms. We try to answer which features are important in the graph (for all nodes). 

There is only one edge mask type `object` available (`shape [E]`), where `E` is the number of directed GROVER bonds. It contains a value for each edge in a GROVER molecular graph. 

### `_get_grover_*_mask`

Two methods `_get_grover_edge_mask` and `_get_grover_node_mask` are available to prepare the edge and node masks, respectively, if available. 

They apply sigmoid to the mask values and add a padding term to the mask. 

### `_loss`

This method computes the loss of the learnable masks, composed by three losses:

1. **Prediction loss:** The mean squared error between the original prediction and the prediction obtained with the current masks. It intends to keep the masked prediction similar to the original.  
2. **Size regularisation:** It adds (edges) or averages (nodes) all the terms in the edge or node mask and multiplies the result by a factor, `edge_size` (Default: 0.005) and `node_feat_size` (Default: 1.0), respectively. It encourages using fewer/smaller masks. 
3. **Entropy regularisation:** It computes the binary entropy function of a mask value $m$ using $H(m)=-mlog(m)-(1-m)log(1-m)$. This function reaches its maximum at $m=0.5$ and it minima at $m\approx0$ and $m\approx1$. Therefore, this loss term encourages mask values close to 0 and 1 instead of around 0.5. The loss term is the average of the binary entropy of all the mask values multiplied by a factor `edge_ent` for the edge mask and `node_feat_ent` for nodes (Default is 1 for both cases).

EXPLICAR HARD EDGE AND NODE MASK

### `_train`

This method sets the GROVER model to evaluation mode, it initialises the edge and node masks and prepares the optimiser. It calculates the original prediction (the target) and optimises the masks in an epoch loop.  

It calculates the `hard_edge_mask` and the `hard_node_mask` looking at the terms with gradient equal to 0 in the first epoch. 

### `_post_process_mask`

It processes the masks by applying the sigmoid and removing attributions for elements that did not receive gradients during the first optimisation step (`hard_*_mask = False`).

### `forward`

It optimises the masks using the method `_train`, the post-processes the masks and returns the node and edge masks using the `GROVERExplanation` class. 

___

The `GROVERExplanation` class





