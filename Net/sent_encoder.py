import torch
import torch.nn as nn
from transformers import BertModel

class PCNN(nn.Module):
    def __init__(self, emb_dim, hidden_size):
        super(PCNN, self).__init__()
        mask_embedding = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        self.mask_embedding = nn.Embedding.from_pretrained(mask_embedding)
        self.cnn = nn.Conv1d(emb_dim, hidden_size, 3, padding=1)
        self.hidden_size = hidden_size
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.cnn.weight)
        nn.init.zeros_(self.cnn.bias)

    def forward(self, X, X_mask):
        X = self.cnn(X.transpose(1, 2)).transpose(1, 2)
        X = self.pool(X, X_mask)
        X = torch.relu(X)
        return X

    def pool(self, X, X_mask):
        X_mask = self.mask_embedding(X_mask)
        X = torch.max(torch.unsqueeze(X_mask, 2) * torch.unsqueeze(X, 3), 1)[0]
        return X.view(-1, self.hidden_size * 3)

class CNN(nn.Module):
    def __init__(self, emb_dim, hidden_size):
        super(CNN, self).__init__()
        self.cnn = nn.Conv1d(emb_dim, hidden_size, 3, padding=1)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.cnn.weight)
        nn.init.zeros_(self.cnn.bias)

    def forward(self, X, _):
        X = self.cnn(X.transpose(1, 2)).transpose(1, 2)
        X, _ = torch.max(X, 1)
        X = torch.relu(X)
        return X

class BERT(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(hidden_size*2, hidden_size)

    def init_weight(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, X, X_Pos1, X_Pos2, X_mask):
        X = self.bert(X, attention_mask=X_mask)
        X = X.last_hidden_state
        # Get entity start hidden state
        onehot_head = torch.zeros(X.shape[:2], dtype=torch.float32, device=X.device)
        onehot_tail = torch.zeros(X.shape[:2], dtype=torch.float32, device=X.device)
        onehot_head = onehot_head.scatter_(1, X_Pos1.unsqueeze(1), 1)
        onehot_tail = onehot_tail.scatter_(1, X_Pos2.unsqueeze(1), 1)
        head_hidden = (onehot_head.unsqueeze(2) * X).sum(1)
        tail_hidden = (onehot_tail.unsqueeze(2) * X).sum(1)
        X = torch.cat([head_hidden, tail_hidden], 1)
        X = self.linear(X)
        return X