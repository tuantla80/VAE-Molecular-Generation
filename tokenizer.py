import numpy as np

# CHARSET is receiced from preprocess.py module
CHARSET = [' ', '#', '(', ')', '+', '-', '/', '1', '2', '3', '4', '5', '6', '7',
           '8', '=', '@', 'B', 'C', 'F', 'H', 'I', 'N', 'O', 'P', 'S', '[', '\\',
           ']', 'c', 'l', 'n', 'o', 'r', 's']


class OneHotTokenizer():

    def __init__(self, charset=CHARSET, fixed_length=120):
        self.charset = charset
        self.fixed_length = fixed_length

    @staticmethod
    def get_one_hot_vector(idx, N):
        '''
        :param idx: index in a vector
        :param N: length of vector
        :return: one hot vector
            Eg. get_one_hot_vector(idx=2, N=6) -> [0, 0, 1, 0, 0, 0]
        '''
        return list(map(int, [idx == i for i in range(N)]))


    @staticmethod
    def get_one_hot_index(chars, char):
        '''
        :param chars: a list of characters or a string
        :param char: a character
        :return: index
            Eg 1: get_one_hot_index(chars=CHARSET,   # CHARSET is a list above
                                    char='#')   -> 1
            Eg 2. get_one_hot_index(chars='Testing',  char='s') -> 2
        '''
        try:
            return chars.index(char)
        except:
            return None


    def pad_smiles(self, smiles):
        '''
        :param smiles: Moleculer SMILES
        :return: smiles with fixed_length
            Eg 1. pad_smiles('banana', fixed_length=20) -> 'banana              '
            Eg 2. pad_smiles('banana', fixed_length=2)  -> 'ba'
            Eg 3. pad_smiles(smiles='COc(c1)cccc1C#N') ->
                  'COc(c1)cccc1C#N                       '  # Total = 120 characters
        '''
        if len(smiles) <= self.fixed_length:
            return smiles.ljust(self.fixed_length)
        return smiles[: self.fixed_length]


    def encode_one_hot(self, smiles):
        '''
        :param smiles: a molecular SMILES
        :return:
        Eg 1. smiles_encode = encode_one_hot(smiles='COc(c1)cccc1C#N')
              Output: smiles_encode =
              [  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  # at position 1 <-> 'C'
                 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0]  # at position 1 <-> 'O'
                 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0]
                 [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
                 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0]
                 ...
                 [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
                 [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
              ]
              with shape = (120, 35)
              120: length of the Smiles (Note: if shorter, make empty values at the end)
              35: each character in smiles is coded in one-hot vector with length = length of CHARSET
        '''
        one_hot_indexes = [self.get_one_hot_index(chars=CHARSET, char=char) for char in self.pad_smiles(smiles)]
        one_hot_vectors = [self.get_one_hot_vector(idx=idx, N=len(CHARSET)) for idx in one_hot_indexes]
        return np.array(one_hot_vectors)


    def tokenize(self, list_smiles):
        return np.array([self.encode_one_hot(smiles) for smiles in list_smiles])


    def decode_one_hot(self, list_encoded_smiles):
        '''
        :param list_encoded_smiles: list of encoded smiles getting from tokenize()
        Eg. list_encoded_smiles =
        [
          # first smiles: 2D array
          [ [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  # first character in smiles. 'C"
            [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0]  # second character in smiles. 'O'
            [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0]
            [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
            [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0]
            ...
            [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  # 119th character in smiles.
            [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  # 120th character in smiles.
          ]
          # second smiles: 2D array
          # third smiles: 2D array
       ]
        -> list_encoded_smiles.shape = (3, 120, 35)   # the first index is number of smiles
        :return: list_smiles_get_back
        '''
        list_smiles_get_back = []
        for smiles_idx in range(len(list_encoded_smiles)):  # run for each smiles
            smiles_string = ''
            for row_idx in range(len(list_encoded_smiles[smiles_idx])):  # run for each row index (each encoded character)
                one_hot = np.argmax(list_encoded_smiles[smiles_idx][row_idx])
                smiles_string += self.charset[one_hot]
            # End of for row_idx
            list_smiles_get_back.append([smiles_string.strip()])
        # End of for smiles_idx
        return list_smiles_get_back


def decode_smiles_from_indexes(vec, charset=CHARSET, decode=True):
    '''
    :param vec: 1D array.
        Eg. vec=np.array([0, 3, 1, 2, 4, 5]
    :param charset: in this code the real charset is np.array
        Eg 1. charset is CHARSET
        charset = [' ', '#', ')', '(', '+', '-', '/', '1', '3', '2', '5',
                   '4', '7', '6', '=', '@', 'C', 'B', 'F', 'I', 'H', 'O',
                   'N', 'S', '[', ']', '\\', 'c', 'l', 'o', 'n', 's', 'r']
        Eg 2. If not decode the charset may have
        charset = [b' ' b'#' b')' b'(' b'+' b'-' b'/' b'1' b'3' b'2' b'5'
                   b'4' b'7' b'6' b'=' b'@' b'C' b'B' b'F' b'I' b'H' b'O'
                   b'N' b'S' b'[' b']' b'\\' b'c' b'l' b'o' b'n' b's' b'r']
    :return:
        Eg 1.  decode_smiles_from_indexes(vec=np.array([0, 3, 1, 2, 4, 5]),
                                          charset='abcdef')
           Since 'abcdef' has indexes 012345,
           Then vec = [0, 3, 1, 2, 4, 5] will generate a string 'adbcef'
    '''
    if decode:
        try:
            charset = np.array([v.decode('utf-8') for v in charset])
        except:
            pass
    # End of if
    return ''.join(map(lambda x: charset[x], vec)).strip()


def test():
    one_hot_tokenizer = OneHotTokenizer(charset=CHARSET, fixed_length=120)
    one_hot_vector = one_hot_tokenizer.get_one_hot_vector(idx=2, N=6)
    print(f'one_hot_vector = {one_hot_vector}')

    one_hot_index = one_hot_tokenizer.get_one_hot_index(chars='Testing', char='s')
    print(f'one_hot_index = {one_hot_index}')

    smiles = one_hot_tokenizer.pad_smiles(smiles='COc(c1)cccc1C#N')
    print(f'smiles = {smiles} with length = {len(smiles)}')

    smiles_encoded = one_hot_tokenizer.encode_one_hot(smiles='COc(c1)cccc1C#N')
    np.set_printoptions(threshold=np.inf)
    print(f'smiles_encoded = {smiles_encoded}')
    print(f'smiles_encoded.shape = {smiles_encoded.shape}')

    list_encoded_smiles = one_hot_tokenizer.tokenize(list_smiles=['COc(c1)cccc1C#N'])
    print(f'\ntokenizer for a list of Smiles = {list_encoded_smiles}')
    print(f'list_encoded_smiles.shape = {list_encoded_smiles.shape}')

    list_smiles_get_back = one_hot_tokenizer.decode_one_hot(list_encoded_smiles)
    print(f'list_smiles_get_back = {list_smiles_get_back}')

    print('\ndecode smiles from indexes - testing with fake smiles')
    fake_smiles = decode_smiles_from_indexes(vec=np.array([0, 3, 1, 2, 4, 5]),
                                             charset='abcdef')
    print(f'fake_smiles = {fake_smiles}')


if __name__ == '__main__':
    test()