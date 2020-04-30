import numpy as np 
import torch
import torch.nn as nn
import torchaudio
import torchvision
import os
import pdb
import matplotlib.pyplot as plt


class Logger(object):
    def __init__(self, model_name):
        # name of model to log        
        self.model_name = model_name
        # lists to track losses
        self.validation_loss = []
        self.training_loss = []
        # lists to track accuracies
        self.validation_accuracy = []
        self.training_accuracy = []

    def log_training(self, batch_loss, batch_acc):
        self.training_loss.append(batch_loss)
        self.training_accuracy.append(batch_acc)

    def log_validation(self, test_loss, test_acc):
        self.validation_loss.append(test_loss)
        self.validation_accuracy.append(test_acc)

    def display_losses(self):
        return 
        plt.figure()
        plt.plot(self.training_loss)
        plt.plot(self.validation_loss)
        plt.show()

    def display_accuracies(self):
        return 
        plt.figure()
        plt.plot(self.training_accuracy)
        plt.plot(self.validation_accuracy)
        plt.show()

        
        
        
        
        
        
        
def learn(epoch, model, trainloader):
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    alpha = 0.5 # center loss weight
    
    for i, data in (enumerate(trainloader)):
        # get input and labels
        x, y = data[0].to(device), data[1].to(device)
        
        # clear out past gradients
        optimizer.zero_grad()
        
        # forward pass
        yhat = model(x)
        ce_loss = criterion(yhat, y)
        
        # combine using centre loss and CE loss
        loss = alpha * center_loss(yhat, y) + ce_loss
        
        # backward step
        loss.backward()
        for param in center_loss.parameters():
            param.grad.data *= (1. / alpha)
        
        # update parameters
        optimizer.step()
        
        _, predicted = torch.max(yhat.data, 1)
        total   += y.size(0)
        correct += (predicted == y).sum().item()
        all_predictions.extend(list(map(lambda t: t.item(), predicted)))
        
        running_loss += loss.item()
        
        logging_interval = 20
        if ((i+1) % logging_interval == 0):
            feedback_string = 'epoch #{:3} \t batch #{:3} \t loss {:.5f} \t accuracy {}'
            print(feedback_string.format(epoch, i+1, running_loss/logging_interval, 100 * correct / total))
            model.logger.log_training(running_loss/logging_interval, 100 * correct / total)
                        
            running_loss = 0.0
            correct = 0
            total = 0
           

        
def validate(epoch, model, dataloader):
    model.eval()
    
    correct = 0
    total = 0
    total_loss = 0.0
    all_predictions = []
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            total_loss += criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(list(map(lambda x: x.item(), predicted)))
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    print('After epoch {}, accuracy of the model is: {:.2f}'.format(epoch, 100 * correct / total))
    model.logger.log_validation(total_loss.item()/total*dataloader.batch_size, 100 * correct / total)
    
    counts = {}
    for i in all_predictions: counts[i] = 1 + counts.get(i, 0)
    print('counts in validation are:', sorted(counts.items()))
    
    

        


