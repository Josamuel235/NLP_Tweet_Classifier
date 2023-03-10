import torch
import torch.nn as nn
import torch.nn.functional as F


class NLP_LSTM(nn.Module):
    def __init__(self, gpu, embedding_dim, hidden_dim, vocab_size, layers, dropout, bi):

        super(NLP_LSTM, self).__init__()
        
        # GPU
        self.gpu = gpu
        
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
        self.fc0 = nn.Linear(embedding_dim, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=layers, batch_first=True, dropout=dropout, bidirectional = bi)
        self.fc1 = nn.Linear(hidden_dim*self.bi, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.drop_layer = nn.Dropout(p=dropout)
        

    def forward(self, inputs):
        
        if self.gpu:
            hidden = torch.zeros(self.layers*self.bi, inputs.shape[0], self.hdim).cuda()
            cell = torch.zeros(self.layers*self.bi, inputs.shape[0], self.hdim).cuda()
        else:
            hidden = torch.zeros(self.layers*self.bi, inputs.shape[0], self.hdim)
            cell = torch.zeros(self.layers*self.bi, inputs.shape[0], self.hdim)
        
        embeddings = self.embedder(inputs)
        linear_layer1 = self.relu(self.fc0(embeddings))
        linear_layer2 = self.relu(self.fc0(linear_layer1))
        dropper1 = self.drop_layer(linear_layer2)
        outputs, (hidden,cell) = self.lstm(dropper1, (hidden,cell))
        linear_layer3 = self.relu(self.fc1(outputs[:,-1,:]))
        dropper2 = self.drop_layer(linear_layer3)
        linear_layer4 = self.fc2(dropper2)
        prediction = torch.sigmoid(linear_layer4)
        return prediction
        
    def tester(self, inputs):
        
        hidden = torch.zeros(self.layers*self.bi, inputs.shape[0], self.hdim).cuda()
        cell = torch.zeros(self.layers*self.bi, inputs.shape[0], self.hdim).cuda()
        
        embeddings = self.embedder(inputs)
        x = self.relu(self.fc0(embeddings))
        y = self.relu(self.fc0(x))
        outputs, (hidden,cell) = self.lstm(y, (hidden,cell))
        linear_layer1 = self.relu(self.fc1(outputs[:,-1,:]))
        linear_layer2 = self.fc2(linear_layer1)
        prediction = torch.sigmoid(linear_layer2)
        return prediction