import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, edge_index, type_num, rel_num, emb_dim, dim1, dim2):
        super(GCN, self).__init__()
        self.type_num = type_num
        self.edge_index = edge_index
        self.node_embedding = nn.Embedding(type_num + rel_num, emb_dim)
        self.conv1 = GCNConv(emb_dim, dim1)
        self.conv2 = GCNConv(dim1, dim2)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.node_embedding.weight)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

    def forward(self):
        X = torch.relu(self.conv1(self.node_embedding.weight, self.edge_index))
        X = torch.relu(self.conv2(X, self.edge_index))
        Type, Rel = X[:self.type_num], X[self.type_num:]
        return Type, Rel