import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0
TOKENIZER = T5TokenizerFast.from_pretrained('google-t5/t5-small')
BOS_ID = TOKENIZER.convert_tokens_to_ids('<extra_id_0>')

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        self.split = split
        self.data = self.process_data(data_folder, split, TOKENIZER)

    def process_data(self, data_folder, split, tokenizer):
        nl = load_lines(os.path.join(data_folder, f'{split}.nl'))
        enc = [tokenizer(q).input_ids for q in nl]
        if split == 'test':
            return [(e, None) for e in enc]
        sql = [s.replace("<", "LESS_THAN") for s in load_lines(os.path.join(data_folder, f'{split}.sql'))]
        dec = [tokenizer(s).input_ids for s in sql]
        return list(zip(enc, dec))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    enc = [torch.tensor(e) for e, _ in batch]
    dec = [torch.tensor(d) for _, d in batch]
    encoder_ids = pad_sequence(enc, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    decoder_targets = pad_sequence(dec, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.full((len(batch), 1), BOS_ID, dtype=torch.long)
    decoder_inputs = torch.cat([initial_decoder_inputs, decoder_targets[:, :-1]], dim=1)
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    enc = [torch.tensor(e) for e, _ in batch]
    encoder_ids = pad_sequence(enc, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.full((len(batch), 1), BOS_ID, dtype=torch.long)
    return encoder_ids, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x