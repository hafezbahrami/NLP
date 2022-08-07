# torchdata is in Beta release and may not be stable. Future API might change
# https://github.com/pytorch/data

"""
The WikiText language modeling dataset is a collection of over 100 million tokens extracted from the set of verified
 Good and Featured articles on Wikipedia. The WikiText-2 dataset is a small version of the WikiText-103 dataset as it
 contains only 2 million tokens.
"""

from typing import Tuple

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import torch
from torch import nn, Tensor
from torch.utils.data import dataset

print_sample_data = False

train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """
    Converts raw text into a flat Tensor.
    Example of using Filter-Tuple:
        aTuple = (2, 5, 8, 1, 14, 3, 16)
        result = filter(lambda x: x % 2 == 0, aTuple)
        result = tuple(result)
        print(result)
    """
    if print_sample_data:
        a = [sent for sent in raw_text_iter]
        print(a[:10])
    # Tokenize each word at each sentence, then convert it to one-hot-vector using vocab object 
    data = [torch.tensor(vocab(tokenizer(sent)), dtype=torch.long) for sent in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data))) # filter our data with zero length, then put all sentences after eachother (torch.cat)

def batchify(data: Tensor, bsz: int, device: torch.device) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size
        device: : torch.device

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz] # making sure to only keep a multiplier of the batch-size. Trim off the rest of sentences.
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)


def get_batch(source: Tensor, i: int, bptt: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int
        bptt: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target



def get_train_test_eval_data(batch_size, eval_batch_size, device: torch.device):
    # train_iter was "consumed" by the process of building the vocab,
    # so we have to create it again
    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter)
    val_data = data_process(val_iter)
    test_data = data_process(test_iter)

    train_data_batched = batchify(train_data, batch_size, device)  # shape [seq_len, batch_size]
    val_data_batched = batchify(val_data, eval_batch_size, device)
    test_data_batched = batchify(test_data, eval_batch_size, device)
    
    return train_data_batched, val_data_batched, test_data_batched




##################################################################
