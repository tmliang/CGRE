import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class GNN(nn.Module):
    def __init__(self, edge_index, type_num, rel_num, emb_dim, emb_mid, emb_out, encoder='gcn', num_layers=2, num_heads=2):
        super(GNN, self).__init__()
        self.type_num = type_num
        self.edge_index = edge_index
        self.node_embedding = nn.Embedding(type_num + rel_num, emb_dim)
        if encoder.lower() == 'gcn':
            self.encoder = nn.ModuleList(
                [
                    GCNConv(emb_dim if i == 0 else emb_mid, emb_mid if i < num_layers - 1 else emb_out)
                    for i in range(num_layers)
                ]
            )
        elif encoder.lower() == "sage":
            self.encoder = nn.ModuleList(
                [
                    SAGEConv(
                        in_channels=emb_dim if i == 0 else emb_mid,
                        out_channels=emb_out if i == num_layers - 1 else emb_mid,
                    )
                    for i in range(num_layers)
                ]
            )

        elif encoder.lower() == "gat":
            assert emb_mid % num_heads == 0
            assert emb_out % num_heads == 0

            self.encoder = nn.ModuleList(
                [
                    GATConv(
                        in_channels=emb_dim if i == 0 else emb_mid,
                        out_channels=(emb_out // num_heads) if i == num_layers - 1 else (emb_mid // num_heads),
                        heads=num_heads,
                        fill_value="mean"
                    )
                    for i in range(num_layers)
                ]
            )
        self.act = nn.GELU()
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.node_embedding.weight)

    def forward(self):
        X = self.node_embedding.weight
        for i, layer in enumerate(self.encoder):
            X = self.act(layer(X, self.edge_index))
        Type, Rel = X[:self.type_num], X[self.type_num:]
        return Type, Rel

