import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
import os
from datetime import datetime

from models import VAE
from tokenizer import OneHotTokenizer


def vae_loss(x_reconstructed, x, z_mean, z_logvar):
    bce_loss = F.binary_cross_entropy(input=x_reconstructed, target=x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return bce_loss + kl_loss


def train(file_path_train_data=r'data\smiles_tokenized.npz',
          path_checkpoint=None,
          checkpoint_save='every',
          file_path_checkpoint_for_continue_learning=None,
          batch_size=1000,
          epochs=100):
    '''
    :param file_path_train_data: file path of data after preprocessing
        Eg. r'data\smiles_tokenized_10000.npz' or r'data\smiles_tokenized.npz'
    :param path_checkpoint:
        if existed, save model check point.
        Eg. checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
    :param checkpoint_save: only activate when path_checkpoint is existed
        checkpoint_save = 'every': save every epoch
                        = 'last': save only final epoch
                        = a number: every 'a number' epoch. eg. every 5 epoch

    :param file_path_checkpoint_for_continue_learning:
        if existed, read the model and optimizer parameters as starting point for training
                    (instead of training from the initial state)
    :param batch_size:
    :param epochs:
    :return:
    '''
    assert checkpoint_save == 'every' or checkpoint_save == 'last' or isinstance(checkpoint_save, int)
    # Get data
    train_data = np.load(file_path_train_data)['arr'].astype(np.float32)  # Eg. X.shape = (249456, 120, 35)
    train_data = torch.utils.data.TensorDataset(torch.from_numpy(train_data))
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Model
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device = {device}')
    model = VAE()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    start_epoch, stop_epoch = 0, epochs
    if file_path_checkpoint_for_continue_learning:
        checkpoint = torch.load(file_path_checkpoint_for_continue_learning, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch, stop_epoch = checkpoint['epoch'] + 1, checkpoint['epoch'] + epochs + 1
    # End of if

    for epoch in range(start_epoch, stop_epoch):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_data_loader):
            data = data[0].to(device)  # Note: data is a list of one 'element'
            optimizer.zero_grad()  # reset - zero out gradient

            # Forward process: compute output (prediction) and loss
            output, z_mean, z_logvar = model(data)
            loss = vae_loss(output, data, z_mean, z_logvar)

            # Backward process: compute gradients
            loss.backward()

            # Update parameters
            optimizer.step()

            # Display some info
            train_loss += loss
            if batch_idx % 100 == 0:
                print(f'\nepoch/batch_idx: {epoch}/{batch_idx}\t loss = {loss: .4f}')
                # Input data
                input_data = data.cpu().numpy()
                print(f'\tFor testing: The first input smiles of batch={batch_size} Smiles')
                print('\t', OneHotTokenizer().decode_one_hot(list_encoded_smiles=[input_data[0]]))
                # Output data
                output_data = output.cpu().detach().numpy()
                print(f'\tFor testing: The first output smiles of {len(output_data)} generated Smiles')
                print('\t', OneHotTokenizer().decode_one_hot(list_encoded_smiles=[output_data[0]]))
        # End of for batch_idx,...
        train_loss /= len(train_data_loader.dataset)
        print(f'Average train loss of this epoch = {train_loss}')

        if path_checkpoint:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            if checkpoint_save == 'every':
                torch.save(obj=checkpoint, f=os.path.join(path_checkpoint, fr'checkpoint_{epoch}.pt'))
            elif checkpoint_save == 'last':
                if epoch == stop_epoch - 1:
                    torch.save(obj=checkpoint, f=os.path.join(path_checkpoint, fr'checkpoint_{epoch}.pt'))
            else:
                if epoch % checkpoint_save == 0:
                    torch.save(obj=checkpoint, f=os.path.join(path_checkpoint, fr'checkpoint_{epoch}.pt'))
        # End of if path_checkpoint:
    # End of for epoch
    return train_loss


if __name__ == '__main__':
    time_start = datetime.now()
    print(f'time_start = {time_start}')
    path = r'data'
    train_loss = train(file_path_train_data=os.path.join(path, 'smiles_tokenized_1000.npz'),
                       path_checkpoint=r'checkpoint',
                       checkpoint_save=2,
                       file_path_checkpoint_for_continue_learning=None,
                       batch_size=1000,
                       epochs=5)
    print(f'train_loss = ', train_loss)
    time_end = datetime.now()
    print(f'time_end = {time_end}')
    print(f'Total running time = {time_end - time_start}')