import torch
import torch.nn as nn

class GROVERExplainer:
    def __init__(
        self,
        model,
        epochs=200,
        lr=0.01,
        edge_size=0.005,
        edge_ent=1.0,
    ):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.edge_size = edge_size
        self.edge_ent = edge_ent

    def explain(self, graph, target):

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = graph

        num_edges = f_bonds.size(0)

        mask_logits = nn.Parameter(
            torch.randn(num_edges, device=f_bonds.device) * 0.1
        )

        optimizer = torch.optim.Adam(
            [mask_logits],
            lr=self.lr
        )

        with torch.no_grad():
            original_prediction = self.model(graph)

        for epoch in range(self.epochs):

            mask = torch.sigmoid(mask_logits)

            prediction = self.model(
                graph,
                edge_mask=mask
            )

            loss = ...

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return mask.detach()




class BondMask(nn.Module):
    def __init__(self, num_bonds):
        super().__init__()
        self.mask_logits = nn.Parameter(torch.randn(num_bonds) * 0.1)

    def forward(self, f_bonds):
        mask = torch.sigmoid(self.mask_logits).unsqueeze(-1)
        return f_bonds * mask, mask

# training loop
mask_module = BondMask(f_bonds.size(0))
opt = torch.optim.Adam(mask_module.parameters(), lr=0.01)
with torch.no_grad():
    orig_pred = grover_model(f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a)

for step in range(200):
    masked_bonds, mask = mask_module(f_bonds)
    pred = grover_model(f_atoms, masked_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a)
    fidelity_loss = F.mse_loss(pred, orig_pred)          # regression (solubility)
    size_loss = mask.mean()
    ent = -(mask * torch.log(mask + 1e-8) + (1 - mask) * torch.log(1 - mask + 1e-8)).mean()
    loss = fidelity_loss + 0.005 * size_loss + 0.01 * ent
    opt.zero_grad(); loss.backward(); opt.step()