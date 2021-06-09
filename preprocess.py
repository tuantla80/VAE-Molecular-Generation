import numpy as np
from datetime import datetime

from tokenizer import OneHotTokenizer


if __name__ == '__main__':
    t0 = datetime.now()
    print(f'Start at = {t0}')
    with open(r'data\zinc_drugs.smi.txt') as f:
        list_smiles = [smiles.strip() for smiles in f]

    one_hot_tokenizer = OneHotTokenizer()
    array_smiled_tokenized = one_hot_tokenizer.tokenize(list_smiles)
    print(f'SHAPE of array_smiled_tokenized = {array_smiled_tokenized.shape}')
    t1 = datetime.now()
    print(f'Finish at = {t1}')

    np.savez_compressed('data\smiles_tokenized.npz', arr=array_smiled_tokenized)

    t2 = datetime.now()
    print(f'Save as npz finish at = {t2}')
