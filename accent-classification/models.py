import numpy as np 
import torch
import torch.nn as nn
import torchaudio
import torchvision
import os
import pdb




class AudioResnet34(torchvision.models.ResNet):
    '''
    CNN for audio classification, with minor modification to ResNet
    '''
    
    def __init__(self, input_channels, num_classes, logger = None):
        # super(AudioResnet34, self).__init__()

        intermediate_size = 50
        super(AudioResnet34, self).__init__(block = torchvision.models.resnet.BasicBlock,
                                           layers = [3, 4, 6, 3],
                                           num_classes = intermediate_size)
        self.conv1 = nn.Conv2d(in_channels = input_channels, out_channels = 64,
                               kernel_size = (7,7), stride = (1, 2), padding = (3,3))
        self.fc1 = nn.Linear(in_features = intermediate_size, out_features = num_classes)
        self.dropout1 = nn.Dropout(p=0.4)
        self.dropout2 = nn.Dropout(p=0.4)
        
        if logger is None:
            raise Exception('Supply a logger object to track loss/acc')
        self.logger = logger
        
    
    def forward(self, x):
#         import pdb; pdb.set_trace();
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.dropout1(x)        
        x = self.fc(x)
        x = self.dropout2(x)
        x = self.fc1(x)

        return x

    def model_name(self):
        return "AudioResnet34"
    
    def save_model(self, optimizer):
        path = './SavedModels/' + self.model_name() 
        path = path + '_epoch' + str(len(self.logger.validation_loss)) 
        path = path + '_accuracy' + str( int(self.logger.validation_accuracy[-1]*100//100) )
        path = path + '.pt'
        
        checkpoint = {'model_state_dict':self.state_dict(), 'optimizer_state_dict':optimizer.state_dict()}
        torch.save(checkpoint, path)

    def load_saved_model(self, saved_model_path, optimizer):
        checkpoint = torch.load(saved_model_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    
    def display_training_graphs(self):
        self.logger.display_losses()

    
    

    
    
class AudioAlexNet(nn.Module):
    def __init__(self):
        '''
        CNN for audio classification, with minor modification to AlexNet
        '''
        super(ExperimentalCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 128,
                               stride = (1, 2), kernel_size = (11, 11),
                               padding = 10, bias = True)
        self.bn1 = nn.BatchNorm2d(num_features = 128)



class SimpleCNN(nn.Module):

    def __init__(self):
        '''
        Simple CNN made up of conv layers followed by dense linear layers
        '''
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = 128,
                               padding=10, stride=2, kernel_size=(201,10))
        self.maxp1 = nn.MaxPool1d(kernel_size = 3, stride = 2, padding = 2)
        self.conv2 = nn.Conv1d(in_channels = 128, out_channels = 256, 
                               stride = 2, kernel_size = 5, padding=4)
        