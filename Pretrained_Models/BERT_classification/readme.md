# Using Pre-trained BERT for classification
#### Dataset

* Loading "ColA" Dataset: a set of sentences labeled as grammatically correct or incorrect. --> Binary Classifications
 We’ll use the wget package to download the dataset to the Colab instance’s file system. The dataset is hosted on GitHub in this repo: https://nyu-mll.github.io/CoLA/
!pip install wget

```
url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'

if not os.path.exists('./cola_public_1.1.zip'):
    wget.download(url, './cola_public_1.1.zip')
```


# Using Pre-trained BERT for Question Answering



