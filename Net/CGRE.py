import torch
import torch.nn as nn
import torch.nn.functional as F
from Net import Entity_Aware_Embedding, PCNN, GCN
import numpy as np


class CGRE(nn.Module):
    def __init__(self, word_vec_dir, edge, constraint, type_num, rel_num, opt):
        super(CGRE, self).__init__()
        word_vec = torch.from_numpy(np.load(word_vec_dir))
        word_dim = word_vec.shape[-1]
        self.rel_num = rel_num
        self.constraint = constraint
        self.Embedding = Entity_Aware_Embedding(word_vec, opt['pos_dim'], opt['max_pos_length'], opt['lambda'])
        self.PCNN = PCNN(word_dim * 3, opt['hidden_size'])
        self.GCN = GCN(edge, type_num, rel_num, opt['graph_emb'], opt['graph_dim'], 3 * opt['hidden_size'])
        self.classifier = nn.Linear(9 * opt['hidden_size'], rel_num)
        self.drop = nn.Dropout(opt['dropout'])
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, X, X_Pos1, X_Pos2, X_Ent1, X_Ent2, X_Mask, X_Type, X_Scope, X_Rel=None):
        Type, Rel = self.GCN()
        X = self.Embedding(X, X_Pos1, X_Pos2, X_Ent1, X_Ent2)
        # Instance representation
        X = self.PCNN(X, X_Mask)
        Type_s = F.embedding(X_Type, Type).reshape(X_Type.shape[0], -1)
        X = torch.cat([X, Type_s], -1)
        # Constraint representation
        Type_r = []
        for i in range(self.rel_num):
            type_rep = F.embedding(self.constraint[i], Type)
            avg_type_rep = type_rep.mean(1)
            Type_r.append(avg_type_rep.reshape(-1))
        Type_r = torch.stack(Type_r)
        Constraints = torch.cat([Rel, Type_r], -1)
        X = self.sentence_attention(X, X_Scope, Constraints, X_Rel)
        return X

    def sentence_attention(self, X, X_Scope, Constraints, X_Rel=None):
        bag_output = []
        if X_Rel is not None:  # For training
            Con = F.embedding(X_Rel, Constraints)
            for i in range(X_Scope.shape[0]):
                bag_rep = X[X_Scope[i][0]: X_Scope[i][1]]
                att_score = F.softmax(bag_rep.matmul(Con[i]), 0).view(1, -1)  # (1, Bag_size)
                att_output = att_score.matmul(bag_rep)  # (1, dim)
                bag_output.append(att_output.squeeze())  # (dim, )
            bag_output = torch.stack(bag_output)
            bag_output = self.drop(bag_output)
            bag_output = self.classifier(bag_output)
        else:  # For testing
            att_score = X.matmul(Constraints.t())  # (Batch_size, dim) -> (Batch_size, R)
            for s in X_Scope:
                bag_rep = X[s[0]:s[1]]  # (Bag_size, dim)
                bag_score = F.softmax(att_score[s[0]:s[1]], 0).t()  # (R, Bag_size)
                att_output = bag_score.matmul(bag_rep)  # (R, dim)
                bag_output.append(torch.diagonal(F.softmax(self.classifier(att_output), -1)))
            bag_output = torch.stack(bag_output)
        return bag_output