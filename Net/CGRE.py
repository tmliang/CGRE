import torch
import torch.nn as nn
import torch.nn.functional as F
from Net import Entity_Aware_Embedding, PCNN, CNN, BERT, GNN
import numpy as np

class CGRE(nn.Module):
    def __init__(self, word_vec_dir, edge, constraint, type_num, rel_num, opt):
        super(CGRE, self).__init__()
        word_vec = torch.from_numpy(np.load(word_vec_dir))
        word_dim = word_vec.shape[-1]
        self.rel_num = rel_num
        self.bert_flag = False
        self.constraint = constraint
        if opt['sent_encoder'].lower() == 'cnn':
            self.Embedding = Entity_Aware_Embedding(word_vec, opt['pos_dim'], opt['max_pos_length'], opt['lambda'])
            self.sent_encoder = CNN(word_dim * 3, opt['hidden_size'])
            sent_out = opt['hidden_size']
        elif opt['sent_encoder'].lower() == 'bert':
            sent_out = 768
            self.bert_flag = True
            self.sent_encoder = BERT(sent_out)
        else: #PCNN
            self.Embedding = Entity_Aware_Embedding(word_vec, opt['pos_dim'], opt['max_pos_length'], opt['lambda'])
            self.sent_encoder = PCNN(word_dim * 3, opt['hidden_size'])
            sent_out = 3 * opt['hidden_size']

        graph_out = opt['graph_out']
        class_dim = opt['class_dim']
        self.fc1 = nn.Linear(sent_out + graph_out * 2, class_dim)
        self.fc2 = nn.Linear(graph_out * 3, class_dim)
        self.classifier = nn.Linear(class_dim, rel_num)
        self.graph_encoder = GNN(edge, type_num, rel_num, opt['graph_emb'], opt['graph_hid'], graph_out,
                                 encoder=opt['graph_encoder'], num_layers=opt['num_layers'], num_heads=opt['num_heads'])
        self.drop = nn.Dropout(opt['dropout'])
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, data):
        X_Rel = None

        # Sentence representation
        if self.bert_flag:
            if len(data) == 7:
                X, X_Pos1, X_Pos2, X_Mask, X_Type, X_Scope, X_Rel = data
            else:
                X, X_Pos1, X_Pos2, X_Mask, X_Type, X_Scope = data
            X = self.sent_encoder(X, X_Pos1, X_Pos2, X_Mask)
        else:
            if len(data) == 10:
                X, X_Pos1, X_Pos2, X_Len, X_Ent1, X_Ent2, X_Mask, X_Type, X_Scope, X_Rel = data
            else:
                X, X_Pos1, X_Pos2, X_Len, X_Ent1, X_Ent2, X_Mask, X_Type, X_Scope = data
            X = self.Embedding(X, X_Pos1, X_Pos2, X_Ent1, X_Ent2)
            X = self.sent_encoder(X, X_Mask)

        # Graph representation
        Type, Rel = self.graph_encoder()

        # Instance representation
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

        X = torch.relu(self.fc1(X))
        Constraints = torch.relu(self.fc2(Constraints))

        # Attention
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