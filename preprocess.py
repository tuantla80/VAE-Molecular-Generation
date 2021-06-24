import numpy as np
from datetime import datetime

from tokenizer import OneHotTokenizer


def get_CHARSET(list_smiles):
    '''To create CHARSET used in tokenizer module
    :param list_smiles: a list of SMILES
    :return: a set of all characters used in SMILES
    '''
    charset = set(' ')  # The first character is ' '
    for smiles in list_smiles:
        for c in smiles:
            charset.add(c)
    # End of for smiles
    return sorted(list(charset))


if __name__ == '__main__':

    t0 = datetime.now()
    print(f'Start at = {t0}')
    with open(r'data\zinc_drugs_250K.smi.txt') as f:
        list_smiles = [smiles.strip() for smiles in f]

    CHARSET = get_CHARSET(list_smiles)
    print(f'CHARSET used in tokenizer.py module: {CHARSET}')

    list_smiles = list_smiles[0:10000]

    one_hot_tokenizer = OneHotTokenizer()
    array_smiled_tokenized = one_hot_tokenizer.tokenize(list_smiles)
    print(f'SHAPE of array_smiled_tokenized = {array_smiled_tokenized.shape}')
    t1 = datetime.now()
    print(f'Finish at = {t1}')

    np.savez_compressed('data\smiles_tokenized_10000.npz', arr=array_smiled_tokenized)

    t2 = datetime.now()
    print(f'Save as npz finish at = {t2}')
