import torch
import torch.nn as nn
import torch.nn.functional as F


class NLP_LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, layers, dropout, bi):

        super(NLP_LSTM, self).__init__()
        
        ## Params
        self.hdim = hidden_dim
        self.layers = layers
        self.drop = dropout
        if bi:
            self.bi = 2
        else:
            self.bi = 1
        
        
        ## layers
        self.embedder = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=layers, batch_first=True, dropout=dropout, bidirectional = bi)
        self.fc1 = nn.Linear(hidden_dim*self.bi, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.drop_layer = nn.Dropout(p=dropout)
        

    def forward(self, inputs):
        
        ## init States
        hidden = torch.zeros(self.layers*self.bi, inputs.shape[0], self.hdim).cuda()
        cell = torch.zeros(self.layers*self.bi, inputs.shape[0], self.hdim).cuda()
        
        ## model forward
        embeddings = self.embedder(inputs)
        outputs, (hidden,cell) = self.lstm(embeddings, (hidden,cell))
        linear_layer1 = self.relu(self.fc1(outputs[:,-1,:]))
        dropper = self.drop_layer(linear_layer1)
        linear_layer2 = self.fc2(dropper)
        prediction = torch.sigmoid(linear_layer2)
        return prediction