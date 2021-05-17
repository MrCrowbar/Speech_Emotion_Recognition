import torch
import sys
import pickle
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from utils import load_data, evaluate, plot_confusion_matrix

from config import model_config as config
from tqdm import tqdm # Se lo agregamos nosotros

class LSTMClassifier(nn.Module):
    """docstring for LSTMClassifier"""
    def __init__(self, config):
        super(LSTMClassifier, self).__init__()
        self.n_layers = config['n_layers']
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.bidirectional = config['bidirectional']
        self.dropout = config['dropout'] if self.n_layers > 1 else 0

        self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, bias=True,
                           num_layers=2, dropout=self.dropout,
                           bidirectional=self.bidirectional)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)
        self.softmax = F.softmax

    def forward(self, input_seq):
        # input_seq =. [1, batch_size, input_size]
        rnn_output, (hidden, _) = self.rnn(input_seq)
        if self.bidirectional:  # sum outputs from the two directions
            rnn_output = rnn_output[:, :, :self.hidden_dim] +\
                        rnn_output[:, :, self.hidden_dim:]
        class_scores = F.softmax(self.out(rnn_output[0]), dim=1)
        return class_scores


if __name__ == '__main__':
    emotion_dict = {'ang': 0, 'hap': 1, 'sad': 2, 'fea': 3, 'sur': 4, 'neu': 5}
    
    # Agregamos variables globales para hacer plot de matrix hasta después de los epochs.
    glob_targets = []
    glob_predictions = []
    
    device = 'cuda:{}'.format(config['gpu']) if torch.cuda.is_available() else 'cpu'
    print("Training on device: ", device) # Le agregamos esto

    model = LSTMClassifier(config)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    train_batches = load_data()
    test_pairs = load_data(test=True)

    best_acc = 0
    for epoch in tqdm(range(config['n_epochs'])): # Le agregamos lo de TQDM
        losses = []
        for batch in train_batches:
            inputs = batch[0].unsqueeze(0)  # frame in format as expected by model
            targets = batch[1]
            inputs = inputs.to(device)
            targets = targets.to(device)

            model.zero_grad()
            optimizer.zero_grad()

            predictions = model(inputs)
            predictions = predictions.to(device)

            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # evaluate
        if epoch % 500 == 0:
            with torch.no_grad():
                inputs = test_pairs[0].unsqueeze(0)
                targets = test_pairs[1]

                inputs = inputs.to(device)
                targets = targets.to(device)

                predictions = torch.argmax(model(inputs), dim=1)  # take argmax to get class id
                predictions = predictions.to(device)

                # evaluate on cpu
                targets = np.array(targets.cpu())
                predictions = np.array(predictions.cpu())
                
                # Agregamos estos arriba
                glob_targets = targets
                glob_predictions = predictions
                
                # Get results
                #plot_confusion_matrix(targets, predictions, classes=emotion_dict.keys())
                performance = evaluate(targets, predictions)
                if performance['acc'] > best_acc:
                    best_acc = performance['acc']
                    print(performance)
                    # save model and results
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, 'runs/{}-best_model.pth'.format(config['model_code']))

                    with open('results/{}-best_performance.pkl'.format(config['model_code']), 'wb') as f:
                        pickle.dump(performance, f)
    plot_confusion_matrix(glob_targets, glob_predictions, classes=emotion_dict.keys()) # Agregamos este al final
