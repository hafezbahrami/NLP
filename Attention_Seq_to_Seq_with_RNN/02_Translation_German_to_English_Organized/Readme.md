# Language Translation (using nn.transformer and torchtext)
We will be using the attention concept, in conjunction with using GRU recurrent-nn model for encoder and decoder.

We are not using the Transformers here.

## 1 Goal: To create a translator from German to English

I)**Preprocess data**: Using torchtext to preprocess data from a well-known dataset containing 
sentences in both English and German and use it to train a sequence-to-sequence model with 
attention that can translate German sentences into English.

We need to preprocess sentences into tensors for NLP modeling and use **torch.utils.data.DataLoader** for training and 
validating the model.

Geart resources are ere: 
 I) On How the Transformers work:
     (a) https://www.youtube.com/watch?v=dichIcUZfOw&ab_channel=MathofIntelligence
     (b) https://www.youtube.com/watch?v=mMa2PmYJlCo&ab_channel=MathofIntelligence
     (c) https://www.youtube.com/watch?v=gJ9kaJsE78k&ab_channel=MathofIntelligence
 
 II) Original Attention All you Need paper: https://nlp.seas.harvard.edu/2018/04/03/attention.html
 III) White-boarding the attention concept:
    https://www.youtube.com/watch?v=yGTUuEx3GkA&ab_channel=Rasa
 IV) Repository with similar task of translation:
     (a) https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb
     (b) https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
 
 
## 2 Data Preprocessing
In general, we need 2 steps: (a) build the vocab object and tokenizing the sentences, and then (b) using DataLoader
library to batch and create an iterator to be used in training-loop for source/target sentences.
  
#### 2-1 Build vocab object
**Why Spicy is preferred over torchtext for translation?**
torchtext has utilities for creating datasets that can be easily iterated through 
for the purposes of creating a language translation model. Here, we will tokenize 
a raw text sentence, build vocabulary, and numericalize tokens into tensor.
```python
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
```

**How to use the vocab objects?**:
```python
en_vocab.get_itos()[20] ==> 'hats'

en_vocab.get_stoi()["hats"] ==> 20
en_vocab["hats"] ==> 20
```

The tokenization uses Spacy. Spacy library is used because it provides strong support
for tokenization in languages other than English. torchtext provides a basic_english 
tokenizer and supports other tokenizers for English (e.g. Moses) but for language 
translation - where multiple languages are required - Spacy is your best bet.
```python
from torchtext.data.utils import get_tokenizer

# German and English tokenizer
de_tokenizer = get_tokenizer('spacy', language='de')
en_tokenizer = get_tokenizer('spacy', language='en')
```

#### 2-2 Download the data set
We should install spacy using pip. Next, download the raw data for the English and 
German Spacy tokenizers:
```python
#pip install spicy
#python -m spacy download en
#python -m spacy download de
#You can now load the package via spacy.load('de_core_news_sm')


from torchtext.utils import download_from_url, extract_archive

url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
train_urls = ('train.de.gz', 'train.en.gz') # for German and English langiages
val_urls = ('val.de.gz', 'val.en.gz')
test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]

```


I got the following message:
```python
Successfully installed en-core-web-sm-3.4.0
Successfully installed de-core-news-sm-3.4.0
```

# 3 Observations worth noting
I) There are 2 sets of parameters (for embedding vector size, encode/decoder/attention dimentions,. ..) in the main.py.
If the code is run with the set with smaller size (such as embedding of only 32, ...)
the result is not impressing. We see the improvement when we change the embedding size from 32 to 256, and ....

II) I am not sure the vocan object I created from torchtext and spacy is free-of-bug.
In process_data.py, for predicting words by index, I wrote the following code, and sometimes I get the
out of the range exception:
```python
def predict_sentence(indexList: List[int], flag="English") -> str:
  try:
    if flag == "English":
      wordList = [en_vocab.get_itos()[idx] for idx in indexList]
    else:
      wordList = [de_vocab.get_itos()[idx] for idx in indexList]
    return " ".join(wordList)
  except:
    pass
    print("Some index are out of the bound. Probably a problem in vocab object")
```