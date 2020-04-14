import torch
import torch.nn as nn


class ATModel(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, batch_size
    ):
        super(ATModel, self).__init__()
        self.batch_size = batch_size
        self.embedding_size = 1024
        self.seq_len = 8
        self.features = int(self.embedding_size / self.seq_len)
        self.hidden_size = int(self.features)

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.conv_fc = nn.Linear(768, self.embedding_size)

        self.lstm = nn.LSTM(
            input_size=self.features,
            hidden_size=self.hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
        )

        self.lstm_fc1 = nn.Linear(self.embedding_size * 2, self.embedding_size)

        self.lstm_fc2 = nn.Linear(self.embedding_size, 128 * 128)

        self.deconv1 = nn.ConvTranspose2d(1, 3, 4, stride=2, padding=1)
        self.bn_dc1 = nn.BatchNorm2d(3)

        self.deconv2 = nn.ConvTranspose2d(3, 1, 4, stride=2, padding=1)
        self.bn_dc2 = nn.BatchNorm2d(1)

    def forward(self, x):
        #         print('input:', x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        #         print('conv1:', x.shape)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        #         print('conv2:', x.shape)

        x = self.maxpool(x)

        #         print('maxpool:', x.shape)
        x = torch.flatten(x, 1)
        x = self.conv_fc(x)
        #         print('conv_fc:', x.shape)
        x = x.view(self.batch_size, self.seq_len, self.features)
        #         print('x_view:', x.shape)
        x, (h_n, c_n) = self.lstm(x)
        #         print('lstm:', x.shape)
        #         x = x.reshape(self.batch_size, 1, 32, 32)
        x = x.reshape(self.batch_size, self.embedding_size * 2)

        #         x = self.deconv1(x)
        #         x = self.bn_dc1(x)
        #         x = self.relu(x)
        #         print('deconv1:', x.shape)

        #         x = self.deconv2(x)
        #         x = self.bn_dc2(x)
        #         x = self.relu(x)
        #         print('deconv2:', x.shape)
        x = self.lstm_fc1(x)
        x = self.relu(x)

        x = self.lstm_fc2(x)
        x = self.relu(x)
        #         print('lstm_fc:', x.shape)
        return x
