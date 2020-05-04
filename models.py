import torch
import torch.nn as nn



class ATModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, batch_size):

        super(ATModel, self).__init__()
        
        # structure 
        self.batch_size = batch_size
        self.embedding_size = 1024
        self.seq_len = 8
        self.features = int(self.embedding_size / self.seq_len)
        self.hidden_size = int(self.features)

        # layers
        # input size: batchsize x chanel size x w x h (the spectrogram)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        

        self.conv_fc = nn.Linear(768, self.embedding_size)

        self.lstm = nn.LSTM(input_size=self.features, 
                            hidden_size=self.hidden_size,
                            num_layers=3,
                            batch_first=True,
                            bidirectional=True)
        self.lstm_fc1 = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.lstm_fc2 = nn.Linear(self.embedding_size, 128 * 128)
        self.deconv1 = nn.ConvTranspose2d(1, 3, 4, stride=2, padding=1) # --> upsample
        self.bn_dc1 = nn.BatchNorm2d(3)
        self.deconv2 = nn.ConvTranspose2d(3, 1, 4, stride=2, padding=1)
        self.bn_dc2 = nn.BatchNorm2d(1)

    def forward(self, x):

        # block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # concat embedding
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        
        ### --->
        #  TODO: ADD CODE TO CONCAT EMBEDDING TO x HERE
        # x = torch.concat(x, )
        ### <---

        # combine feature map with embedding
        x = self.conv_fc(x) # TODO: CHANGE CONV_FC DIMENSIONS TO ACCOUNT CONCAT WITH EMBEDDING
        
        # reshape and take through LSTM
        x = x.view(self.batch_size, self.seq_len, self.features)
        x, (h_n, c_n) = self.lstm(x)

        x = x.reshape(self.batch_size, self.embedding_size * 2)

        # TODO: ASK SHUBY WHY THE DECONV WAS REMOVED
        x = self.lstm_fc1(x)
        x = self.relu(x)

        x = self.lstm_fc2(x)
        x = self.relu(x)

        return x
