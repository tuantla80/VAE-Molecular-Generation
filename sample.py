import numpy as np
import torch
from torch import optim

from models import VAE
from tokenizer import OneHotTokenizer

file_path_chk = r'checkpoint\checkpoint_100.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = VAE()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
checkpoint = torch.load(file_path_chk, map_location=device)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])

# Input data
smiles = 'O=Cc1ccc(O)c(OC)c1' # It is Vanillin
one_hot_tokenizer = OneHotTokenizer()
smiles_encoded = one_hot_tokenizer.encode_one_hot(smiles=smiles)  # numpy.ndarray, shape=(120, 35)
smiles_encoded = torch.Tensor([smiles_encoded])
print(f'type(smiles_encoded = {type(smiles_encoded)}, smiles_encoded.shape={smiles_encoded.shape}')
smiles_encoded = smiles_encoded.to(device)

# Output
recon_encoded = model(smiles_encoded) # Note at model.py, output is y, z_mean, z_logvar
print(f'type(recon_encoded) = {type(recon_encoded)}') # Tuple of (y, z_mean, z_logvar)
smiles_recon_encoded = recon_encoded[0].cpu().detach().numpy()
print(f'smiles_recon_encoded.shape = {smiles_recon_encoded.shape}')  # (1, 120, 35)
smiles_recon = one_hot_tokenizer.decode_one_hot(list_encoded_smiles=smiles_recon_encoded)

z_mean = recon_encoded[1].cpu().detach().numpy()
z_logvar = recon_encoded[2].cpu().detach().numpy()
print(f'\nz_mean.shape = {z_mean.shape}, z_logvar.shape={z_logvar.shape}')
z_mean = torch.Tensor(z_mean)
z_logvar = torch.Tensor(z_logvar)

# Generate one Smiles
std = torch.exp(0.5 * z_logvar)
epsilon = 0.01 * torch.randn_like(input=std)  # multiply 1e-2 to make epsilon smaller
# z = z_mean + std * epsilon
z = 1.5 * z_mean + 3 * std * epsilon
y = model.decode(z) # print(f'y.shape = {y.shape}')  # y = VAE().decode(z)
y_0 = y[0].cpu().detach().numpy()

gen_smiles = one_hot_tokenizer.decode_one_hot(list_encoded_smiles=[y_0])

print(f'Input SMILES  = {smiles}')
print(f'Output SMILES = {smiles_recon[0][0]}')
print(f'Example of generated smiles = {gen_smiles[0][0]}')

'''
Output example:

C:\ProgramData\Anaconda3\python.exe C:/AI/VAE-Molecular-Generation/sample.py
type(smiles_encoded = <class 'torch.Tensor'>, smiles_encoded.shape=torch.Size([1, 120, 35])
type(recon_encoded) = <class 'tuple'>
smiles_recon_encoded.shape = (1, 120, 35)

z_mean.shape = (1, 292), z_logvar.shape=(1, 292)
Input SMILES  = O=Cc1ccc(O)c(OC)c1
Output SMILES = O=Cc1ccc(O)c(OC)c1
Example of generated smiles = O=Oc1cccc1O
'''
