import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F


class VAE(nn.Module):
    '''
    Input data:
        Shape = (batch, 120, 35)
    '''
    def __init__(self):
        super(VAE, self).__init__()

        self.conv_1 = nn.Conv1d(in_channels=120, out_channels=9, kernel_size=9, stride=1)
        self.conv_2 = nn.Conv1d(in_channels=9, out_channels=9, kernel_size=9, stride=1)
        self.conv_3 = nn.Conv1d(in_channels=9, out_channels=10, kernel_size=11, stride=1)

        self.fc_0 = nn.Linear(in_features=90, out_features=435)
        self.fc_1 = nn.Linear(in_features=435, out_features=292)
        self.fc_2 = nn.Linear(in_features=435, out_features=292)
        self.fc_3 = nn.Linear(in_features=292, out_features=292)

        self.gru = nn.GRU(input_size=292, hidden_size=501, num_layers=3, batch_first=True)
        self.fc_4 = nn.Linear(in_features=501, out_features=35)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()


    def encode(self, x):
        '''
        :param x:
        :return:
        Example
        import numpy
        import torch.nn as nn
        import torch.nn.functional as F
        import torch

        batch_size = 64
        x = torch.rand(batch_size, 120, 35)

        # Convolutional layer
        x = F.relu(nn.Conv1d(120, 9, kernel_size=9)(x))      # x.shape=torch.Size([64, 9, 27])
        x = F.relu(nn.Conv1d(9, 9, kernel_size=9)(x))        # x.shape=torch.Size([64, 9, 19])
        x = F.relu(nn.Conv1d(9, 10, kernel_size=11)(x))      # x.shape=torch.Size([64, 10, 9])

        # fatten 2 last dimensions but keep the batch_size
        x = x.view(x.size(0), -1)                            # x.shape=torch.Size([64, 90])

        # Fully connected layer
        x = F.selu(nn.Linear(90, 435)(x))                    # x.shape=torch.Size([64, 435])

        # Get z_mean and z_logvar (log-variance)
        z_mean = nn.Linear(435, 292)(x)                      # x.shape=torch.Size([64, 292])
        z_logvar = nn.Linear(435, 292)(x)                    # x.shape=torch.Size([64, 292])
        '''
        # Convolutional layer
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))

        # Fatten 2 last dimension but keep the batch_size
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = F.selu(self.fc_0(x))

        # return z_mean and z_logvar
        return self.fc_1(x), self.fc_2(x)


    def sampling(self, z_mean, z_logvar):
        '''
        It is a parameterization trick to sample to get latent variable Z
        :param z_mean: an output tensor of a standard fully connected layer from encoder (rf. encode() function)
        :param z_logvar: an output tensor of a standard fully connected layer from encoder (rf. encode() function)
        :return: z (latent variable)
            z = z_mean + std * epsilon

        Note. torch.randn_like(input): Returns a tensor with the same size as input that
              is filled with random numbers from a normal distribution with mean 0 and
              variance 1. Therefore, input here is just to get shape.

        Example: continue with example in encode() method. Note: 64 is batch_size
        std = torch.exp(0.5 * z_logvar)               # std.shape=torch.Size([64, 292])
        epsilon = 1e-2 * torch.randn_like(input=std)  # epsilon.shape=torch.Size([64, 292])
        z = z_mean + std * epsilon                    # z.shape=torch.Size([64, 292])
        '''
        std = torch.exp(0.5 * z_logvar)
        epsilon = 1e-2 * torch.randn_like(input=std)  # multiply 1e-2 to make epsilon smaller
        return  z_mean + std * epsilon


    def decode(self, z):
        '''
        :param z:
        :return:

        Example: continue with example in sampling() method
        z = F.selu(nn.Linear(292, 292)(z))                      # z.shape=torch.Size([64, 292])
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 120, 1)  # z.shape=torch.Size([64, 120, 292])
        output, h_n = nn.GRU(292, 501,
                             num_layers=3,
                             batch_first=True)(z)               # output.shape=torch.Size([64, 120, 501])
                                                                # h_n.shape=torch.Size([3, 64, 501])
        out_reshape = output.contiguous()
                            .view(-1, output.size(-1))          # out_reshape=torch.Size([7680, 501]) # 7680=64*120

        y_out = nn.Linear(501, 35)(out_reshape)                 # y_out.shape=torch.Size([7680, 35])
        y_out = F.softmax(y_out, dim=1)                         # y_out.shape=torch.Size([7680, 35])
                                                                # dim=1 -> sum to 1 to every row
        y = y_out.contiguous()
                 .view(output.size(0), -1, y_out.size(-1))      # y.shape=torch.Size([64, 120, 35])
        '''
        z = F.selu(self.fc_3(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 120, 1)
        output, h_n = self.gru(z)
        output_reshape = output.contiguous().view(-1, output.size(-1))
        y_out = F.softmax(self.fc_4(output_reshape), dim=1)
        y = y_out.contiguous().view(output.size(0), -1, y_out.size(-1))
        return y
    
    
    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar)
        y = self.decode(z)
        return y, z_mean, z_logvar


def test_class_VAE():
    batch = 64
    inputs = torch.rand(batch, 120, 35)
    y, z_mean, z_logvar = VAE().forward(x=inputs)
    print(f'output: y.shape = {y.shape}')
    print(f'latent space: z_mean.shape = {z_mean.shape}')
    print(f'latent space: z_logvar.shape = {z_logvar.shape}')
        

if __name__ == '__main__':
    print('Run a test for forward VAE')
    test_class_VAE()