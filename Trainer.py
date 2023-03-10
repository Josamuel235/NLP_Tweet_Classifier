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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report


class Trainer:
    def __init__(self, filepath, models, batch_size, emded_dim, hidden_dim, learning_rate, epochs, gam, train_size, test_size, early, layers, thresh, most, dropout, bi):

        self.gpu_avail = torch.cuda.is_available()
        loaded_Data = Data_Creator(filepath)
        self.vocabulary = vocab_builder(loaded_Data.tokenized_df)
        processed_data = data_processor(loaded_Data.tokenized_df, self.vocabulary, test_size, train_size, thresh, most)
        
        ## Data
        self.train_data = processed_data.train
        self.vlad_data = processed_data.validate
        self.test_data = processed_data.test

        ## Data Loaders
        self.train_loader = DataLoader(Dataset(self.train_data), batch_size=batch_size, shuffle=True)
        self.vlad_loader = DataLoader(Dataset(self.vlad_data), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(Dataset(self.test_data), batch_size=batch_size, shuffle=True)

        ## Models
        self.models = models

        ## Params
        self.last_epoch = epochs
        self.early = early
        self.bs = batch_size

        ## model
        if self.gpu_avail:
            self.loss = nn.BCELoss().cuda()
            self.model = NLP_LSTM(self.gpu_avail, emded_dim, hidden_dim, len(self.vocabulary.word_idx), layers, dropout, bi).cuda().float()
            self.best = deepcopy(self.model.state_dict())
        else: 
            self.loss = torch.nn.BCELoss()
            self.model = NLP_LSTM(self.gpu_avail, emded_dim, hidden_dim, len(self.vocabulary.word_idx), layers, dropout, bi)

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
        if 'simple' in self.models:
            simple = self.Simple_gusser_Model()
            print('The Simple Evaluation Metrics:\n')
            print(simple)
            print("\n")
        if 'svm' in self.models:
            svm = self.svm_model()
            print('The SVM Evaluation Metrics:\n')
            print(svm)
            print("\n")
        if 'lstm' in self.models:
            lstm = self.run_LSTM()
            print('The LSTM Evaluation Metrics:\n')
            print(lstm)
            print("\n")


    def run_LSTM(self):
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
        print(end_time - start_time)
        self.model.load_state_dict(self.best)
        self.model.train(False)
        self.save_model()
        self.plot_stats()
        return self.test()

    
    def train(self):
        epoch_loss = []
        correct = 0
        total = 0
        
        for text, tag in self.train_loader:
            if self.gpu_avail:
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
            if self.gpu_avail:
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
        pred =[]
        actual = []
        correct = 0 
        total = 0
        
        for text, tag in self.test_loader:
            if self.gpu_avail:
                text = text.cuda()
                tag = tag.cuda()
            outputs = self.model.tester(text)
            loss = self.loss(outputs, tag)
            epoch_loss.append(loss.item())
            classification = torch.round(outputs.squeeze())
            pred.extend(classification.cpu().detach().numpy())
            actual.extend(tag.squeeze().cpu().detach().numpy())
            num_correct = torch.eq(classification, tag.squeeze()).squeeze()
            correct += torch.sum(num_correct)
            total += (tag.squeeze()).size(0)
            
        acc = correct/total
        avg = np.array(epoch_loss).mean()
        print(F"test_loss: {avg}")
        print(F"test accuracy: {acc}")
        return classification_report(actual, pred, output_dict=True)
        
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


    # SVM model
    def svm_model(self):
        vectorizer = TfidfVectorizer(min_df = 5,
                                 max_df = 0.8,
                                 stop_words = 'english',
                                 sublinear_tf = True,
                                 use_idf = True)

        def convert(x):
            keep = []
            for i in x:
                keep.append(self.vocabulary.idx_word[i])
            return ' '.join(keep)

        ## Pre-process
        svm_train = self.train_data['numerized_tweet'].apply(lambda x: convert(x))
        svm_test = self.test_data['numerized_tweet'].apply(lambda x: convert(x))
        
        #TD-IDF
        X_train = vectorizer.fit_transform(svm_train)
        X_test = vectorizer.transform(svm_test)
        
        # Labels
        y_train = self.train_data['Toxicity']
        y_test = self.test_data['Toxicity']
        
        
        #SVM 
        classifier = svm.LinearSVC(C = 10**-2)
        
        
        #SVM Train
        classifier.fit(X_train, y_train)
        
        #SVM Test
        predictions = classifier.predict(X_test)
        
        return classification_report(y_test, predictions, output_dict=True)
    

    # Simple model
    def Simple_gusser_Model(self):

        np.random.choice([1,0], p =[0.45, 0.55])
        
        base = []
        for i in range(len(self.test_data['Toxicity'])):
            base.append(np.random.choice([1,0], p =[0.45, 0.55]))
        
        return classification_report(self.test_data['Toxicity'], base, output_dict=True)