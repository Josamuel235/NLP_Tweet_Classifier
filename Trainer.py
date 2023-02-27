import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from copy import deepcopy
import matplotlib.pyplot as plt
from Vocabulary import vocab_builder
from Dataset import Dataset
from Data_Creator import Data_Creator
from Data_Processor import data_processor
from LSTM_Architecture import NLP_LSTM


class Trainer:
    def __init__(self, filepath, batch_size, emded_dim, hidden_dim, learning_rate, epochs, gam, train_size, test_size, early, layers, thresh, most, dropout, bi):

        self.gpu_avail = torch.cuda.is_available()
        loaded_Data = Data_Creator(filepath)
        self.vocabulary = vocab_builder(loaded_Data.tokenized_df)
        processed_data = data_processor(loaded_Data.tokenized_df, self.vocabulary, test_size, train_size, thresh, most)
        
        ## Data
        train_data = processed_data.train
        vlad_data = processed_data.validate
        test_data = processed_data.test

        ## Data Loaders

        self.train_loader = DataLoader(Dataset(train_data), batch_size=batch_size, shuffle=True)
        self.vlad_loader = DataLoader(Dataset(vlad_data), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(Dataset(test_data), batch_size=batch_size, shuffle=True)

        ## Params
        self.last_epoch = epochs
        self.early = early
        self.bs = batch_size

        ## model
        if self.gpu_avail:
            self.loss = nn.BCELoss().cuda()
            self.model = NLP_LSTM(emded_dim, hidden_dim, len(self.vocabulary.word_idx), layers, dropout, bi).cuda().float()
            self.best = deepcopy(self.model.state_dict())
        else: 
            self.loss = torch.nn.BCELoss()
            self.model = NLP_LSTM(emded_dim, hidden_dim, len(self.vocabulary.word_idx))

        ## Optimizer    
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gam)

        ## Stat Tracking
        self.min_loss = float('inf')
        
        self.train_loss = []
        self.vlad_loss = []
        
        self.train_acc = []
        self.vlad_acc = []


    def run(self):
        print("Training Commencing")
        start_time = datetime.now()

        for epoch in range(0, self.last_epoch):
            print(F'epoch: {epoch+1}')
            print('Training\n')
            self.model.train(True)
            self.train()
            print('Testing\n')
            self.model.train(False)
            self.validation()
            if self.vlad_loss[-1] < self.min_loss:
                count=0
                self.min_loss = self.vlad_loss[-1]
                self.best = deepcopy(self.model.state_dict())
            if count> self.early:
                print('Early Stopping\n')
                break
            count += 1

        end_time = datetime.now()
        print("Training time: " + str(end_time-start_time))
        self.model.load_state_dict(self.best)
        self.model.train(False)
        self.save_model()
        self.plot_stats()
        self.test()

    
    def train(self):
        epoch_loss = []
        correct = 0
        total = 0
        
        for text, tag in self.train_loader:
            text = text.cuda()
            tag = tag.cuda()
            self.optimizer.zero_grad()
            outputs = self.model.forward(text)
            loss = self.loss(outputs, tag)
            loss.backward()
            self.optimizer.step()
            epoch_loss.append(loss.item())
            classification = torch.round(outputs.squeeze())
            num_correct = torch.eq(classification, tag.squeeze()).squeeze()
            correct += torch.sum(num_correct)
            total += (tag.squeeze()).size(0)

        self.lr_scheduler.step()
        avg = np.array(epoch_loss).mean()
        self.train_loss.append(avg)
        acc = correct/total
        self.train_acc.append(acc)
        print(F"train loss: {avg}")
        print(F"train accuracy: {acc}")


    def validation(self):
        epoch_loss = []
        correct = 0
        total = 0
        
        for text, tag in self.vlad_loader:
            text = text.cuda()
            tag = tag.cuda()
            outputs = self.model.forward(text)
            loss = self.loss(outputs, tag)
            epoch_loss.append(loss.item())
            classification = torch.round(outputs.squeeze())
            num_correct = torch.eq(classification, tag.squeeze()).squeeze()
            correct += torch.sum(num_correct)
            total += (tag.squeeze()).size(0)
            
        acc = correct/total
        avg = np.array(epoch_loss).mean()
        self.vlad_loss.append(avg)
        self.vlad_acc.append(acc)
        print(F"validation loss: {avg}")
        print(F"validation accuracy: {acc}")

    def test(self):
        epoch_loss = []
        correct = 0 
        total = 0
        
        for text, tag in self.test_loader:
            text = text.cuda()
            tag = tag.cuda()
            outputs = self.model.forward(text)
            loss = self.loss(outputs, tag)
            epoch_loss.append(loss.item())
            classification = torch.round(outputs.squeeze())
            num_correct = torch.eq(classification, tag.squeeze()).squeeze()
            correct += torch.sum(num_correct)
            total += (tag.squeeze()).size(0)
            
        acc = correct/total
        avg = np.array(epoch_loss).mean()
        print(F"test_loss: {avg}")
        print(F"test accuracy: {acc}")
        
    def save_model(self):
        model_path = 'latest_model.pt'
        model_dict = self.model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.optimizer.state_dict()}
        torch.save(state_dict, model_path)

    def plot_stats(self):
        e = len(self.train_loss)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure(figsize=(10,6))
        plt.plot(x_axis, self.train_loss, label="Training Loss")
        plt.plot(x_axis, self.vlad_loss, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title("LSTM Sentiment" + " Stats Plot")
        plt.savefig("Loss_plot.png")
        plt.show()
        plt.figure(figsize=(10,6))
        plt.plot(x_axis, self.train_loss, label="Training Acc")
        plt.plot(x_axis, self.vlad_loss, label="Validation Acc")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title("LSTM Sentiment" + " Stats Plot")
        plt.savefig("Accuracy_plot.png")
        plt.show()
