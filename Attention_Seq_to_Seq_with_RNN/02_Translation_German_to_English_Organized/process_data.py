from typing import List
import torchtext
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab
from torchtext.utils import download_from_url, extract_archive
import io


from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# -------------------------------------------------------------------------------
# Step1: torchtext and spacy to create vocab object and tokenize the sentences

url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
train_urls = ('train.de.gz', 'train.en.gz') # for German and English langiages
val_urls = ('val.de.gz', 'val.en.gz')
test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]

# German and English tokenizer
de_tokenizer = get_tokenizer('spacy', language='de')
en_tokenizer = get_tokenizer('spacy', language='en')
# run the following on Google Colab
# de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
# en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

def build_vocab(filepath, tokenizer):
  counter = Counter()
  with io.open(filepath, encoding="utf8") as f:
    for string_ in f:
      counter.update(tokenizer(string_))
  return vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

# Build the vocab object, and set special indexes
de_vocab = build_vocab(train_filepaths[0], de_tokenizer)
en_vocab = build_vocab(train_filepaths[1], en_tokenizer)
for item in ['<unk>', '<pad>', '<bos>', '<eos>']:
  de_vocab.set_default_index(de_vocab[item])
  en_vocab.set_default_index(en_vocab[item])


PAD_IDX = de_vocab['<pad>'] # index for pad
BOS_IDX = de_vocab['<bos>'] # index for beginning of sentence
EOS_IDX = de_vocab['<eos>'] # index for end of sentences

def _data_process(filepaths):
  raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
  raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
  data = []
  for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
    de_tensor_ = torch.tensor([de_vocab[token] for token in de_tokenizer(raw_de)], dtype=torch.long) # example: de_vocab["Buchen"]=3
    en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en)], dtype=torch.long) # example: en_vocab["book"]=1130
    data.append((de_tensor_, en_tensor_))
  return data

# -------------------------------------------------------------------------------
# Step2: Use Data-loader library

def _generate_batch(data_batch):
  de_batch, en_batch = [], []
  for (de_item, en_item) in data_batch:
    de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
    en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
  de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
  en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
  return de_batch, en_batch


def get_data_and_vocab(BATCH_SIZE):
  # first, let's tokenize the sentences (using torchtext and spacy)
  train_data = _data_process(train_filepaths) # passing the train data path for both English and German. train_data[0][0] for German and train_data[0][1] for English
  val_data = _data_process(val_filepaths)
  test_data = _data_process(test_filepaths)

  # using DataLoader to shuffle and batch the data
  train_iter = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=_generate_batch)
  valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=_generate_batch)
  test_iter = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=_generate_batch)

  return en_vocab, de_vocab, train_iter, valid_iter, test_iter

# --------------------------------------------------------------------
# Predict a sentence from index
def predict_sentence(indexList: List[int], flag="English") -> str:
  try:
    if flag == "English":
      wordList = [en_vocab.get_itos()[idx] for idx in indexList]
    else:
      wordList = [de_vocab.get_itos()[idx] for idx in indexList]
    return " ".join(wordList)
  except:
    pass
    print("Some index are out of the bound. Probbaly a problem in vocab object")
