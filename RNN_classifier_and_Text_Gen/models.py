# models.py

from typing import List
import random
import numpy as np
import collections
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import time
import math
from utils import Indexer


# from torch.utils.tensorboard import SummaryWriter #to print to tensorboard

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

category_list = ["consonant", "vowel"] # not the best practice. Just bc not wanted to pass the whole dataloader around

#####################
# PART 0: Frequency based classifications - easiest and just to get familiar with the process
#####################

class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")


class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """

    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1

def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)


#####################
# MODELS FOR PART 1 #
#####################


class RNNClassifier(ConsonantVowelClassifier):
    def __init__(self, rnn_trained_model):
        self.rnn_trained_model = rnn_trained_model

    def predict(self, context):
        seq_lengths = len(context)
        input, _ = str2ascii_arr(context)
        input = torch.as_tensor(input)
        input.unsqueeze_(0)
        output, _ = self.rnn_trained_model(input)
        pred = output.data.max(1, keepdim=True)[1]
        return pred.item()

# Batch Building
class PhraseDataset(Dataset):
    """ dataset."""

    def __init__(self, class_vowel_data: List[str], class_consonant_data: List[str]):
        # assuming two classes has almost the same number of elements
        list1 = [(phrase, "vowel") for phrase in class_vowel_data]
        list2 = [(phrase, "consonant") for phrase in class_consonant_data]
        rows = list1 + list2
        random.shuffle(rows)
        self.phrases = [row[0] for row in rows]
        self.category = [row[1] for row in rows]
        self.len = len(self.phrases)

        self.category_list = list(sorted(set(self.category)))

    def __getitem__(self, index):
        return self.phrases[index], self.category[index]

    def __len__(self):
        return self.len

    def get_categories(self):
        return self.category_list

    def get_category(self, id):
        return self.category_list[id]

    def get_category_id(self, category):
        return self.category_list.index(category)

    def create_variable(tensor):
        # Do cuda() before wrapping with variable
        if torch.cuda.is_available():
            return Variable(tensor.cuda())
        else:
            return Variable(tensor)

def create_variable(tensor):
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

def categories2tensor(categories):
    categories_ids = []
    for c in categories:
        categories_ids.append(category_list.index(c))
    return torch.LongTensor(categories_ids)

# pad sequences and sort the tensor
# if the words are not of the same length, the input ASCII will be the length of the max word length, and the
# rest of shorter words will get zero padding. For this case that we are solving, all the phrases are of the same length.
def pad_sequences(vectorized_seqs, seq_lengths, categories):
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # Sort tensors by their length
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]

    # Also sort the target (countries) in the same order
    target = categories2tensor(categories)
    if len(categories):
        target = target[perm_idx]

    # Return variables
    # DataParallel requires everything to be a Variable
    return create_variable(seq_tensor), \
            create_variable(seq_lengths), \
            create_variable(target)

def str2ascii_arr(msg):
    arr = [ord(c) for c in msg]
    return arr, len(arr)

# Create necessary variables, lengths, and target
def make_variables(phrases, categories):
    sequence_and_length = [str2ascii_arr(ph) for ph in phrases]
    vectorized_seqs = [sl[0] for sl in sequence_and_length]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])
    return pad_sequences(vectorized_seqs, seq_lengths, categories)

# Some utility functions
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class RnnModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, dropout=0.2, bidirectional=True):
        super(RnnModel, self).__init__()
        self.input_size = input_size #vector size, as input into embedding layer
        self.hidden_size = hidden_size # The vector size as hidden state
        self.num_layers = num_layers
        self.output_size = num_classes
        self.n_directions = int(bidirectional) + 1
        # layer 1: embedding layer
        self.word_embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=hidden_size)
        # Assume the vector size of the input to GRU and the hidden state are the same
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=False, bidirectional=bidirectional)

        self.fc = nn.Linear(self.hidden_size, self.output_size) # from output of LSTM layer to the actual output size

        self.init_weight()

    def init_weight(self):
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def _init_hidden_state(self, batch_size):
        hidden = torch.zeros(self.num_layers * self.n_directions,
        batch_size, self.hidden_size)
        return create_variable(hidden)


    def forward(self, input):
        # Input X is basically an array of ASCII integers, corresponding to the each char of the entire word

        # input shape: B(batch) x S ---> transpose to make S(sequence) x B(batch)
        input = input.t()
        batch_size = input.size(1)

        # Make a hidden
        hidden = self._init_hidden_state(batch_size)

        # Embedding S x B -> S x B x I (embedding size)
        embedded = self.word_embedding(input)

        output, hidden = self.gru(embedded, hidden)
        # In the last cell, the hidden and the output are identical, or use the output instead. Since the output
        # contains all the output for all cells, we need to pick the one comes from the last cell
        fc_classification_output = self.fc(hidden[-1])  # for classification purpose. Only use the last output cell
        fc_text_gen_output = self.fc(output) # for text generation. Use all output cells
        return fc_classification_output, fc_text_gen_output


def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index) -> RNNClassifier:
    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """
    n_epocs = 60
    batch_size = 150
    sequence_length = len(vocab_index)
    hidden_size = 100
    n_char = 128 # number of ASCII numbers of all 26 characters
    num_classes = 2 # binary classifications

    # Preparing Dataset
    train_dataset = PhraseDataset(class_vowel_data=train_vowel_exs, class_consonant_data=train_cons_exs)
    train_dataset_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = PhraseDataset(class_vowel_data=dev_vowel_exs, class_consonant_data=dev_cons_exs)
    test_dataset_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


    classifier_model = RnnModel(input_size=n_char, hidden_size=hidden_size, num_classes=num_classes, num_layers = 1, bidirectional=False).to(device)
    classifier_model.train()


    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    loss_per_epoc = []
    start = time.time()
    print("Training for {} epochs...".format(n_epocs))
    for epoch in range(1, n_epocs + 1):
        # Train cycle
        total_loss = 0

        for i, (phrases, categories) in enumerate(train_dataset_loader, 1):
            input, seq_lengths, target = make_variables(phrases, categories)
            output, _ = classifier_model(input)

            loss = criterion(output, target)
            total_loss += loss.item()

            classifier_model.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f}'.format(
                time_since(start), epoch, i *
                len(phrases), len(train_dataset_loader.dataset),
                100. * i * len(phrases) / len(train_dataset_loader.dataset),
                total_loss / i * len(phrases)))
                loss_per_epoc.append(total_loss)

    # Testing
    if epoch%5 == 0:
        print("evaluating trained model ...")
        correct = 0
        train_data_size = len(test_dataset_loader.dataset)

        for phrases, categories in test_dataset_loader:
            input, seq_lengths, target = make_variables(phrases, categories)
            output, _ = classifier_model(input)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, train_data_size, 100. * correct / train_data_size))

    return RNNClassifier(classifier_model)


#####################################
# MODELS FOR PART 2 # Text Generation
#####################################

def _get_index_from_a_text(text: str, vocab_index) -> List[int]:
    return [vocab_index.index_of(c) for c in text]

def _get_text_from_a_vocab_index(list_int: List[int], vocab_index) -> str:
    res = [vocab_index.get_object(i) for i in list_int]
    return "".join(res)

def _create_text_chunks_in_index(text: str, chunk_len: int, vocab_index):
    """if the text=abcdef , and chunk_len=3, then the X=abc, bcd, cde,
                                                 label=bcd, cde, def """
    text_in_index = _get_index_from_a_text(text, vocab_index)
    text_len = len(text)
    x = []
    label = []
    for i in range(text_len):
        if (i + chunk_len) < text_len:
            x.append(text_in_index[i: i + chunk_len])
            label.append(text_in_index[i + 1: i + chunk_len + 1])
    return x, label


def _get_text_in_vocab_index(text, vocab_index):
    """Revert back the target values from ASCII to the word_index which is limited to 27 small-case letter [a, b,c, ....]"""
    for i, x in enumerate(text.numpy()):
        for j, y in enumerate(x):
            text[i, j] = vocab_index.index_of(chr(y))
    return text

# Batch Building
class TextGenDataset:
    """batching the input and labels for text generation"""

    def __init__(self, x_text: List[int], y_text: List[int]):
        self.x_text = torch.as_tensor(x_text, dtype=int)
        self.y_text = torch.as_tensor(y_text, dtype=int)

    def __getitem__(self, index):
        return self.x_text[index], self.y_text[index]

    def __len__(self):
        return len(self.x_text)


def make_variables_for_text_gen(x_texts, y_texts, vocab_index):
    n_sentences = len(x_texts)
    n_char = len(x_texts[0])
    x_seq = torch.zeros((n_sentences, n_char)).long()
    y_seq = torch.zeros((n_sentences, n_char)).long()
    for i in range(n_sentences):
        for j in range(n_char):
            x_seq[i][j] = vocab_index.index_of(x_texts[i][j])
            y_seq[i][j] = vocab_index.index_of(y_texts[i][j])
    return x_seq, y_seq


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e
        :param context: a single character to score
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0 / self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0 / self.voc_size) * len(next_chars)


class RNNLanguageModel(LanguageModel):
    def __init__(self, lm_model: RnnModel, vocab_index: Indexer):
        self.lm_model = lm_model
        self.vocab_index = vocab_index

    def get_next_char_log_probs(self, context):
        log_softmax_last_cell = self._get_log_softmax_last_cell(context)
        return np.float64(log_softmax_last_cell)

    def get_log_prob_sequence(self, next_chars, context):
        log_softmax_last_cell = self._get_log_softmax_last_cell(context)
        idx = self.vocab_index.index_of(next_chars)
        return np.float64(log_softmax_last_cell[idx])

    def _get_log_softmax_last_cell(self, context):
        indexed_text = _get_index_from_a_text(context, self.vocab_index)
        indexed_text = torch.as_tensor(indexed_text, dtype=int)
        indexed_text = indexed_text.view(1, -1) # --> (batch=1, chunk)
        _, output = self.lm_model(indexed_text)
        last_cell_output = output[len(context)-1, 0, :] # only get the output of the linear layer corresponding to the last cell
        softmax_output = nn.Softmax(dim=0)(last_cell_output)
        return np.log(softmax_output.detach().numpy())

def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    chunk_len = 40
    if chunk_len > len(dev_text):
        chunk_len = len(dev_text) - 1
    n_epocs = 3
    batch_size = 150
    sequence_length = len(vocab_index)
    hidden_size = 50
    # n_char = 128  # number of ASCII numbers of all 26 characters
    n_char = len(vocab_index)  # Eliminate the ASCII numbers layer (as it is not necessary)
    num_classes = len(vocab_index)  # each output from the last linear layer can predict one out of 27 character

    x_train, label_train = _create_text_chunks_in_index(train_text, chunk_len, vocab_index)
    x_test, label_test = _create_text_chunks_in_index(dev_text, chunk_len, vocab_index)

    # Preparing Dataset
    train_text_dataset = TextGenDataset(x_train, label_train)
    train_text_dataset_loader = DataLoader(dataset=train_text_dataset, batch_size=batch_size, shuffle=True)
    test_text_dataset = TextGenDataset(x_test, label_test)
    test_text_dataset_loader = DataLoader(dataset=test_text_dataset, batch_size=batch_size, shuffle=True)

    text_gen_model = RnnModel(input_size=n_char, hidden_size=hidden_size, num_classes=num_classes, num_layers=2,
                              bidirectional=False).to(device)
    text_gen_model.train()

    optimizer = torch.optim.Adam(text_gen_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    loss_per_epoc = []
    start = time.time()
    print("Training for {} epochs...".format(n_epocs))
    for epoch in range(1, n_epocs + 1):
        # Train cycle
        total_loss = 0

        for i, (x_text, label_text) in enumerate(train_text_dataset_loader, 1):
            # getting x_text and label_text into ASCII code (for this practice we might not need, as we are limited to small-case letters only)
            # x_text, label_text = make_variables_for_text_gen(x_text, label_text, vocab_index)

            # label as --> (chunk, Batch)
            label_index = label_text.t()
            # output --> (chunk, batch, num_classes)
            _, output = text_gen_model(x_text)

            for c in range(chunk_len):
                loss = criterion(output[c, :, :], label_index[c, :])
                total_loss += loss.item()

            text_gen_model.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f}'.format(
                    time_since(start), epoch, i *
                                              len(x_text), len(train_text_dataset_loader.dataset),
                                              100. * i * len(x_text) / len(train_text_dataset_loader.dataset),
                                              total_loss / i * len(x_text)))
        loss_per_epoc.append(total_loss)

        # Testing
        if epoch % 1 == 0:
            print("evaluating trained model ...")
            correct = 0
            test_data_len = len(test_text_dataset_loader.dataset)

            for x_text_test, label_text_test in test_text_dataset_loader:
                # x_text_test, label_text_test = make_variables_for_text_gen(x_text_test, label_text_test)
                # label_index_test = _get_text_in_vocab_index(label_text_test, vocab_index)
                label_index_test = label_text_test.t()
                _, output = text_gen_model(x_text_test)

                for c in range(chunk_len):
                    pred = output[c, :, :].data.max(1, keepdim=True)[1]
                    correct += pred.eq(label_index_test[c, :].data.view_as(pred)).cpu().sum()

            print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
                correct, test_data_len*chunk_len, 100. * correct / (test_data_len*chunk_len)))
    return RNNLanguageModel(text_gen_model, vocab_index)
