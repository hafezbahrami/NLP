import random
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils import *
from data import *
from lf_evaluator import *
from typing import List
import argparse

# ---------------------------------------------------------------------------------------------------
# Attention Plot
import io
import PIL.Image
import torchvision
from torchvision.transforms import ToTensor
import matplotlib
from matplotlib import pyplot as plt
# # ---------------------------------------------------------------------------------------------------
# # -------------------------------------------------------------------------------------------------------------
# # TensorBoard stuff
# import torch.utils.tensorboard as tb
#
# log_dir = "./log/" # This log directory should be consistent with the one defined in GoogleColab in "%tensorboard --logdir . --port 6006"
# tensor_board_logger = tb.SummaryWriter(log_dir, flush_secs=1)
# # -------------------------------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add_models_args(parser):
    """
    Command-line arguments to the system related to your model. Feel free to extend here.
    """
    # Some common arguments for your convenience
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=30, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--n_layers', type=int, default=1, help="number of layers in RNN for encoder and decoder.")
    parser.add_argument('--clip_the_gradient', type=float, default=50., help='In order to prevent over-shoting in the gradient')

    parser.add_argument('--embed_size', type=int, default=100, help='embedding vector size')
    parser.add_argument('--hidden_size', type=int, default=50, help='hidden cell size')
    parser.add_argument('--embedding_dropout', type=float, default=0.2, help='drop out ratio in embedding layer')
    parser.add_argument('--encdec_dropout', type=float, default=0.1, help='drop out for encoder and decoder RNN')
    parser.add_argument('--bidirectional', default=True, action='store_true', help="Bidirectional RNN (GRU or LSTM)")
    # Do not change the batchfirst to False. SOme of internal methods are based on Batch first order
    parser.add_argument('--batchfirst', default=True, action='store_true', help="Batch_First flag for RNN (GRU or LSTM)")
    parser.add_argument('--attentionmethod', type=str, default="general", help="Function for attention method: genral, dot, concat.")
    # 65 is all you need for GeoQuery
    parser.add_argument('--decoder_len_limit', type=int, default=65, help='output length limit of the decoder')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0, help="Method for updating the inputs for decoder.")
    parser.add_argument('--beam_search_length', type=int, default=1, help='Length of the beam in beam search.')


    parser.add_argument('--attention_decoder', default=True, action='store_true', help="Whether or not to implement the attention to the decoder.")
    parser.add_argument('--loss_plot_frequency', type=int, default=1, help='Frequency of epoch to plot the loss.')
    parser.add_argument('--model_evaluation_frequency', type=int, default=1, help='Frequency of epoch to evaluate the model.')
    parser.add_argument('--attention_plot', default=False, action='store_true', help="Whether or not plot the attention matrix")

    # Feel free to add other hyperparameters for your input dimension, etc. to control your network
    # 50-200 might be a good range to start with for embedding and LSTM sizes

    args = parser.parse_args()
    return args


# Only for testing the code. It is not an actual parser.
class NearestNeighborSemanticParser(object):
    """
    Semantic parser that uses Jaccard similarity to find the most similar input example to a particular question and
    returns the associated logical form.
    """
    def __init__(self, training_data: List[Example]):
        self.training_data = training_data

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        """
        :param test_data: List[Example] to decode
        :return: A list of k-best lists of Derivations. A Derivation consists of the underlying Example, a probability,
        and a tokenized input string. If you're just doing one-best decoding of example ex and you
        produce output y_tok, you can just return the k-best list [Derivation(ex, 1.0, y_tok)]
        """
        test_derivs = []
        for test_ex in test_data:
            test_words = test_ex.x_tok
            best_jaccard = -1
            best_train_ex = None
            # Find the highest word overlap with the train data
            for train_ex in self.training_data:
                # Compute word overlap with Jaccard similarity
                train_words = train_ex.x_tok
                overlap = len(frozenset(train_words) & frozenset(test_words))
                jaccard = overlap/float(len(frozenset(train_words) | frozenset(test_words)))
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_train_ex = train_ex
            # Note that this is a list of a single Derivation
            test_derivs.append([Derivation(test_ex, 1.0, best_train_ex.y_tok)])
        return test_derivs

class EmbeddingLayer(nn.Module):
    """
    Embedding layer that has a lookup table of symbols that is [full_dict_size x input_dim]. Includes dropout.
    Works for both non-batched and batched inputs
    """
    # Parameters: dimension of the word embeddings, number of words, and the dropout rate to apply
    # (0.2 is often a reasonable value)
    def __init__(self, input_dim: int, full_dict_size: int, embedding_dropout_rate: float):
        """
        :param input_dim: dimensionality of the word vectors
        :param full_dict_size: number of words in the vocabulary
        :param embedding_dropout_rate: dropout rate to apply
        """
        super(EmbeddingLayer, self).__init__()
        self.dropout = nn.Dropout(embedding_dropout_rate)
        self.word_embedding = nn.Embedding(full_dict_size, input_dim)

    # TODO: delete this part
    def add_pretrained(self, word_vectors):
        self.word_embedding = self.word_embedding.from_pretrained(word_vectors, freeze = False)

    # Takes either a non-batched input [sent len x input_dim] or a batched input
    # [batch size x sent len x input dim]
    def forward(self, input):
        """
        :param input: either a non-batched input [sent len x voc size] or a batched input
        [batch size x sent len x voc size]
        :return: embedded form of the input words (last coordinate replaced by input_dim)
        """
        embedded_words = self.word_embedding(input)
        final_embeddings = self.dropout(embedded_words)
        return final_embeddings

class RNNEncoder(nn.Module):
    def __init__(self, model_args):
        super(RNNEncoder, self).__init__()
        self.bidirect = model_args.bidirectional
        self.batch_first = model_args.batchfirst
        self.n_layer = model_args.n_layers
        self.input_size = model_args.embed_size
        self.hidden_size = model_args.hidden_size
        self.dropout = model_args.encdec_dropout

        self.reduce_h_W = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.reduce_c_W = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.n_layer, batch_first=self.batch_first,
                                                        dropout=self.dropout, bidirectional=self.bidirect)
        self.init_weight()

    # Initializes weight matrices using Xavier initialization
    def init_weight(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        if self.bidirect:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0_reverse, gain=1)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0_reverse, gain=1)
            nn.init.constant_(self.rnn.bias_hh_l0, 0)
            nn.init.constant_(self.rnn.bias_ih_l0, 0)
            if self.bidirect:
                nn.init.constant_(self.rnn.bias_hh_l0_reverse, 0)
                nn.init.constant_(self.rnn.bias_ih_l0_reverse, 0)

    def get_output_size(self):
        return self.hidden_size * 2 if self.bidirect else self.hidden_size

    def sent_lens_to_mask(self, lens, max_length):
        return torch.from_numpy(np.asarray
                ([[1 if j < lens.data[i].item() else 0 for j in range(0, max_length)] for i in range(0, lens.shape[0])]))

    def forward(self, embedded_words, input_lens):
        # Takes the embedded sentences, "packs" them into an efficient Pytorch-internal representation
        packed_embedding = nn.utils.rnn.pack_padded_sequence(embedded_words, input_lens.cpu(), batch_first=self.batch_first, enforce_sorted=False)
        # Runs the RNN over each sequence. Returns output at each position as well as the last vectors of the RNN
        # state for each sentence (first/last vectors for bidirectional)
        output, hn = self.rnn(packed_embedding)
        # Unpacks the Pytorch representation into normal tensors
        output, sent_lens = nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        max_length = input_lens.data[0].item()
        context_mask = self.sent_lens_to_mask(sent_lens, max_length)
         
        if self.bidirect: # the following bidirectional part is assuming we only have 1 layer in our LSTM
            h, c = hn[0], hn[1] # hn = (hidden state, cell state) --> h: 2*n_layer X B X hidden_sie & c has the same size as h
            # h[0] and h[1] represent the hidden state in 2 directions. We just basically concat 2 directions --> h_: h: B X 2*hidden_sie
            h_, c_ = torch.cat((h[0], h[1]), dim=1), torch.cat((c[0], c[1]), dim=1)
            # as the hidden size in the encoder
            bi_ht = (h_, c_) # hidden and state concatted for 2 directions
            new_h = self.reduce_h_W(h_)
            new_c = self.reduce_c_W(c_)
            h_t = (new_h, new_c) # Reduce the bi-direction effects, to make them ready for decoder and the attention
        else:
            h, c = hn[0][0], hn[1][0]
            h_t = (h, c)
        return (output, context_mask, h_t, bi_ht) # h_t as reduced dimenstion to eliminate the birectionality. and bi_ht contains the concat version of both hidden and cell states


class RNNDecoder(nn.Module):
    # Parameters: input size (should match embedding layer), hidden size for the LSTM, dropout rate for the RNN,
    # and a boolean flag for whether or not we're using a bidirectional encoder
    def __init__(self, out, model_args):
        super(RNNDecoder, self).__init__()
        self.input_size = model_args.embed_size
        self.hidden_size = model_args.hidden_size
        self.n_layer = model_args.n_layers
        self.dropout = 0 # model_args.encdec_dropout
        self.out = out
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.n_layer,
                           batch_first=model_args.batchfirst, dropout=self.dropout)
        self.ff = nn.Linear(self.hidden_size, out)
        self.log_softmax = nn.LogSoftmax(dim = 1)

        self.init_weight()

    # Initializes weight matrices using Xavier initialization
    def init_weight(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        nn.init.xavier_uniform_(self.ff.weight)


    # Like the encoder, the embedding layer is outside the decoder. So, it is assumed here
    # that the input is a word embedding
    def forward(self, input, hidden):
        # print("input: ", input)
        output, (h ,c) = self.rnn(input, hidden)
        output = self.ff(h[0])
        return self.log_softmax(output), (h ,c)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, num_output_classes, model_args):
        super(LuongAttnDecoderRNN, self).__init__()
        self.dropout = model_args.encdec_dropout
        self.n_layers = model_args.n_layers
        self.batchfirst = model_args.batchfirst
        self.bidirect = model_args.bidirectional
        self.n_directions = int(self.bidirect) + 1
        # In Attention, we concat the output of the embedding (embed_vec) with the context vecctor (which is the
        # bi-directional hidden state from previous cell or from encoder)
        self.input_size = model_args.embed_size + (self.n_directions * model_args.hidden_size)
        self.hidden_size_enc = self.n_directions * model_args.hidden_size
        self.hidden_size_dec = model_args.hidden_size
        self.out = num_output_classes

        self.rnn = nn.LSTM(self.input_size, self.hidden_size_dec, num_layers=self.n_layers,
                                    batch_first=self.batchfirst, dropout=self.dropout)
        self.attn = nn.Linear(self.hidden_size_enc, self.hidden_size_dec) # to convert the potential bidirectional output to uni-directional
        self.attn_hid = nn.Linear(self.hidden_size_dec + self.hidden_size_enc, self.hidden_size_dec) # After concatenating the context vector ci and the hidden state hbar, get it ready for final classification layer
        self.classifier = nn.Linear(self.hidden_size_dec, self.out) # classification layer
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.init_weight()


    # Initializes weight matrices using Xavier initialization
    def init_weight(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        nn.init.xavier_uniform_(self.attn.weight)
        nn.init.xavier_uniform_(self.attn_hid.weight)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, input, hidden, encoder_outputs):
        output, (h, c) = self.rnn(input, hidden)  # At this point output is identical to h --> h == output
        h_bar = h[0]  # Assuming 1-layer LSTM pick the hidden-state(B X hidden_size) for attention

        encoder_outputs = encoder_outputs.squeeze()  # Remove the Batch dimension: encoder_seq_len X (2*hidden_size)

        # Calculate eij
        attn_weight = self.attn(
            encoder_outputs).squeeze()  # get the potential directional output, and make it like uni-directional: : encoder_seq_len X hidden_size
        attn_weight = torch.transpose(attn_weight, 0,
                                      1)  # hidden_size X encoder_seq_len --> to make it suitable for matrix multiplication
        attn_energy = torch.matmul(h_bar, attn_weight)  # B X encoder_seq_len
        attn_score = F.softmax(attn_energy, dim=1)  # eij: B X encoder_seq_len

        # context vector (ci)
        context = torch.matmul(attn_score, encoder_outputs)  # Calculate the context vector, ci

        # making dimensions ready for final classifications
        attn_hid_combined = torch.cat((context, h_bar),
                                      1)  # Concatenate the context vector ci and the hidden state hbar
        attn_hid_transformed = self.attn_hid(attn_hid_combined)
        out = self.classifier(attn_hid_transformed)

        return self.log_softmax(out), (h, c), context, attn_score


class Seq2SeqSemanticParser(object):
    def __init__(self, model_input_emb, model_enc, model_output_emb, model_dec, model_args, output_indexer):
        self.model_input_emb = model_input_emb
        self.model_enc = model_enc
        self.model_output_emb = model_output_emb
        self.model_dec = model_dec
        self.model_args = model_args
        self.output_indexer = output_indexer

    def decode(self, test_data, attention_plot=False):
        beam_length = self.model_args.beam_search_length
        if beam_length > 1:  # if beam search is required, switch to the other decoding method
            return self.decode_beam(self, test_data)

        self.model_input_emb.eval()
        self.model_enc.eval()
        self.model_output_emb.eval()
        self.model_dec.eval()

        self.model_input_emb.zero_grad()
        self.model_output_emb.zero_grad()
        self.model_enc.zero_grad()
        self.model_dec.zero_grad()

        SOS_token = self.output_indexer.index_of('<SOS>')
        EOS_token = self.output_indexer.index_of('<EOS>')
        derivations = []


        for ex in test_data:
            y_toks = []
            self.model_input_emb.zero_grad()
            self.model_output_emb.zero_grad()
            self.model_enc.zero_grad()
            self.model_dec.zero_grad()

            x_tensor = torch.as_tensor([ex.x_indexed]).to(device)
            y_tensor = torch.as_tensor([ex.y_indexed]).to(device)
            inp_lens_tensor = torch.as_tensor([len(ex.x_indexed)]).to(device)

            enc_output, enc_context_mask, enc_hidden, enc_bi_hidden = _helper_encode_input_for_decoder(
                                                        x_tensor, inp_lens_tensor, self.model_input_emb, self.model_enc)
            dec_hidden = enc_hidden

            context_vec = enc_bi_hidden[0]
            dec_input = torch.as_tensor([[SOS_token]]).to(device)
            y_temp = []

            # only for attention plotting purpose
            att_scores = torch.zeros(self.model_args.decoder_len_limit + 1, self.model_args.decoder_len_limit + 1).to(device)
            #dec_cell_log_prob_output, dec_selected_cls_idx, dec_selected_cls_log_prob, dec_cell_hidden, context_vec, attn_score
            count = 0
            while (dec_input.item() != EOS_token) and count <= self.model_args.decoder_len_limit:
                if self.model_args.attention_decoder: # with attention
                    dec_cell_log_prob_output, dec_selected_cls_idx, dec_selected_cls_log_prob, dec_cell_hidden, context_vec, attn_score = _helper_decode_attn(
                        dec_input, dec_hidden,
                        self.model_output_emb, self.model_dec,
                        context_vec,
                        enc_output, beam_length)
                    dec_input = dec_selected_cls_idx
                    dec_hidden = dec_cell_hidden

                    att_scores[count, :attn_score.size(1)] += attn_score.squeeze(0).data
                else:
                    dec_output, dec_input, dec_input_val, dec_hidden = _helper_decode(
                        dec_input, dec_hidden,
                        self.model_output_emb, self.model_dec,
                        beam_length)
                y_label = self.output_indexer.get_object(dec_input.item())
                if dec_input.item() != EOS_token:
                    y_toks.append(y_label)
                    y_temp.append(dec_input.item())
                count = count + 1
            derivations.append([Derivation(ex, 1.0, y_toks)])
            if self.model_args.attention_decoder and attention_plot:
                non_zero_att_scores = att_scores[:count+1, :enc_output.size(1)]
                self.show_attention(ex.x_tok, y_toks, non_zero_att_scores)

        return derivations


    def decode_beam(self, test_data):
        test_data.to(device)
        print("Decoding using beam-search-length > 1.")
        self.model_input_emb.eval()
        self.model_enc.eval()
        self.model_output_emb.eval()
        self.model_dec.eval()
        self.model_input_emb.zero_grad()
        self.model_output_emb.zero_grad()
        self.model_enc.zero_grad()
        self.model_dec.zero_grad()

        SOS_token = self.output_indexer.index_of('<SOS>')
        EOS_token = self.output_indexer.index_of('<EOS>')
        beam_length = self.model_args.beam_search_length
        derivations = []

        beam = Beam_v2(beam_length) # Beam_v2 class is a little modified compared to the original Beam class

        ex_count = 0

        for ex in test_data:
            count = 0
            ex_derivs = []
            y_toks = []
            self.model_input_emb.zero_grad()
            self.model_output_emb.zero_grad()
            self.model_enc.zero_grad()
            self.model_dec.zero_grad()

            x_tensor = torch.as_tensor([ex.x_indexed]).to(device)
            y_tensor = torch.as_tensor([ex.y_indexed]).to(device)
            inp_lens_tensor = torch.as_tensor([len(ex.x_indexed)]).to(device)

            enc_output, enc_context_mask, enc_hidden, enc_bi_hidden = _helper_encode_input_for_decoder(
                x_tensor, inp_lens_tensor, self.model_input_emb, self.model_enc)
            dec_hidden = enc_hidden

            context_vec = enc_bi_hidden[0]
            dec_input = torch.as_tensor([[SOS_token]]).to(device)
            y_temp = []
            beam.add(dec_input, 0.0, dec_hidden, [])

            while count <= self.args.decoder_len_limit:
                beam_temp = Beam_v2(beam_length)
                # Check if all dec_input items are EOS, if so, break.
                if beam.all_EOS(EOS_token):
                    break

                for dec_input, dec_prob, dec_hidden, path in beam.get_elts_and_scores():
                    # print("dec_input: ", dec_input)
                    if dec_input.item() == EOS_token:
                        beam_temp.add(dec_input, dec_prob, dec_hidden, path)
                    if dec_input.item() != EOS_token:
                        if self.model_args.attention_decoder:  # with attention
                            dec_cell_log_prob_output, dec_selected_cls_idx, dec_selected_cls_log_prob, dec_cell_hidden, context_vec, attn_score = _helper_decode_attn(
                                dec_input, dec_hidden,
                                self.model_output_emb, self.model_dec,
                                context_vec,
                                enc_output, beam_length)
                            dec_input = dec_selected_cls_idx
                            dec_hidden = dec_cell_hidden
                        else: # with no attention
                            dec_output, dec_input, dec_input_val, dec_hidden = _helper_decode(
                                dec_input, dec_hidden,
                                self.model_output_emb, self.model_dec,
                                beam_length)

                        dec_input = dec_input.squeeze()
                        dec_input_val = dec_input_val.squeeze()
                        for i in range(len(dec_input)):
                            y_label = self.output_indexer.get_object(dec_input[i].item())
                            dec_prob_temp = (dec_prob + dec_input_val[i].item()) / (len(path) + 1)
                            beam_temp.add(dec_input[i].unsqueeze(0).unsqueeze(0), dec_prob_temp, dec_hidden,
                                          path + [y_label])

                beam = beam_temp
                count = count + 1
            ex_count = ex_count + 1

            print("True path: ", ex.x_indexed)
            for dec_input, dec_prob, dec_hidden, path in beam.get_elts_and_scores():
                ex_derivs.append(Derivation(ex, dec_prob, path))
                print("Predicted path: ", path)

            derivations.append(ex_derivs)
        return derivations

    def show_attention(self, input_sentence, output_words, attentions):
        # Set up figure with colorbar
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.cpu().detach().numpy(), cmap='bone')
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([''] + input_sentence + ['<EOS>'], rotation=90)
        ax.set_yticklabels([''] + output_words)

        # Show label at every tick
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))

        # Save the pyplot plot object to the buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png') # saving the plot into buffer
        buf.seek(0)

        plot_buf = buf

        image = PIL.Image.open(plot_buf)
        image = ToTensor()(image)

        # tensor_board_logger.add_image("attention_score_box", image) # show in tensor_board in GoogleColab
        # plt.show()
        # plt.close()


#-------------------------------------------------------------------------------
# helper functions and classes
#-------------------------------------------------------------------------------

def sequence_mask(sequence_length, max_len=None):
    """ For instance, if lens=[10, 6, 8], the max_length must be = 10. Then, it returns (good for loss calculation):
    [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]
    Replace 1's with Truw, and 0's with False
    """
    if max_len is None:
        max_len = sequence_length.data.max()

    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand).to(device)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """
    length = length.to(device)

    logits_flat = logits.view(-1, logits.size(-1)) # logits_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=1) # log_probs_flat: (batch * max_len, num_classes)
    target_flat = target.view(-1, 1) # target_flat: (batch * max_len, 1)
    target_flat = target_flat.type(torch.LongTensor).to(device) # making sure target_flat is in Long type

    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)  # losses_flat: (batch * max_len, 1)
    losses = losses_flat.view(*target.size()) # losses: (batch, max_len)

    mask = sequence_mask(sequence_length=length, max_len=target.size(1)) # mask: (batch, max_len)
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss


def make_padded_input_tensor(exs: List[Example], input_indexer: Indexer, max_len: int, reverse_input=False) -> np.ndarray:
    """
    Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len.
    Optionally reverses them.
    :param exs: examples to tensor-ify
    :param input_indexer: Indexer over input symbols; needed to get the index of the pad symbol
    :param max_len: max input len to use (pad/truncate to this length)
    :param reverse_input: True if we should reverse the inputs (useful if doing a unidirectional LSTM encoder)
    :return: A [num example, max_len]-size array of indices of the input tokens
    """

    if reverse_input:
        return np.array(
            [[ex.x_indexed[len(ex.x_indexed) - 1 - i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.x_indexed[i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])

def make_padded_output_tensor(exs, output_indexer, max_len):
    """
    Similar to make_padded_input_tensor, but does it on the outputs without the option to reverse input
    :param exs:
    :param output_indexer:
    :param max_len:
    :return: A [num example, max_len]-size array of indices of the output tokens
    """
    return np.array \
        ([[ex.y_indexed[i] if i < len(ex.y_indexed) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)] for ex in exs])


class CoderDecoderDataset(Dataset):
    """For Batching the Coder Decoder Dataset"""
    def __init__(self, data, input_indexer, output_indexer, reverse_input=False):
        input_actual_lengths = [len(s.x_indexed) for s in data]
        input_max_len = np.max(input_actual_lengths)
        all_input_data = make_padded_input_tensor(data, input_indexer, input_max_len, reverse_input=reverse_input)

        output_actual_lengths = [len(s.y_indexed) for s in data]
        output_max_len = np.max(output_actual_lengths)
        all_output_data = make_padded_output_tensor(data, output_indexer, output_max_len)

        self.x_sentences = all_input_data
        self.input_actual_lengths = input_actual_lengths
        self.x_max_sentence_len = input_max_len

        self.y_sentences = all_output_data
        self.output_actual_lengths = output_actual_lengths
        self.y_max_sentence_len = output_max_len

        self.len = len(self.x_sentences)

        if self.len != len(self.y_sentences):
            raise Exception

    def __getitem__(self, index):
        return self.x_sentences[index], self.input_actual_lengths[index], self.x_max_sentence_len, self.y_sentences[index], self.output_actual_lengths[index], self.y_max_sentence_len

    def __len__(self):
        return self.len

def _helper_decode(y_index, hidden, model_output_emb, model_dec, beam_length):
    output_emb = model_output_emb.forward(y_index)
    output, hidden = model_dec.forward(output_emb, hidden)
    top_val, top_idx = output.topk(beam_length)
    dec_input_log_prob = top_val.detach()
    dec_input_idx = top_idx.detach()
    return output, dec_input_idx, dec_input_log_prob, hidden

def _helper_decode_attn(y_index, hidden, model_output_emb, dec_model, context_vec, encoder_output, beam_length):
    embedded_dec_input = model_output_emb.forward(y_index)
    embedded_dec_input_and_context= torch.cat((embedded_dec_input.squeeze(1), context_vec), dim = 1) # We concat the context vector with the embedded input
    embedded_dec_input_and_context = embedded_dec_input_and_context.unsqueeze(0)
    dec_cell_log_prob_output, dec_cell_hidden, context_vec, attn_score = dec_model.forward(embedded_dec_input_and_context, hidden, encoder_output)
    top_val, top_idx = dec_cell_log_prob_output.topk(beam_length)
    dec_selected_cls_idx = top_idx.detach()
    dec_selected_cls_log_prob = top_val.detach()
    return dec_cell_log_prob_output, dec_selected_cls_idx, dec_selected_cls_log_prob, dec_cell_hidden, context_vec, attn_score


def _helper_encode_input_for_decoder(x_tensor, inp_lens_tensor, model_input_emb, model_enc):
    input_emb = model_input_emb.forward(x_tensor)
    (enc_output_each_word, enc_context_mask, enc_final_states, enc_bi_hidden) = model_enc.forward(input_emb, inp_lens_tensor)
    enc_final_states_reshaped = \
    (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))  # (hidden-state , cell-state)
    return (enc_output_each_word, enc_context_mask, enc_final_states_reshaped, enc_bi_hidden)


def train_for_a_batch_of_data(x_sens, x_sens_lens, y_sens, y_sens_lens,
                                model_input_emb, model_enc, model_output_emb, model_dec,
                                enc_optimizer, dec_optimizer, input_emb_optimizer, output_emb_optimizer,
                                input_indexer, output_indexer,
                                model_args):

    # Set all models to training mode
    model_input_emb.train()
    model_enc.train()
    model_output_emb.train()
    model_dec.train()

    # Initialize loss
    loss = 0
    SOS_token = output_indexer.index_of("<SOS>")
    EOS_token = output_indexer.index_of("<EOS>")
    beam_search_length = model_args.beam_search_length
    criterion = torch.nn.NLLLoss()

    model_input_emb.zero_grad()
    model_output_emb.zero_grad()
    model_enc.zero_grad()
    model_dec.zero_grad()

    x_tensor = x_sens
    y_tensor = y_sens
    inp_lens_tensor = x_sens_lens

    enc_output, enc_context_mask, enc_hidden, enc_bi_hidden = _helper_encode_input_for_decoder(
                                                                x_tensor, inp_lens_tensor,
                                                                model_input_emb, model_enc)

    dec_hidden = enc_hidden
    context_vec = enc_bi_hidden[0] # enc_bi_hidden tuple contains birectional hidden-state and cell-state. We pick the bidirectional hidden state as the context vector

    dec_input = torch.as_tensor([[SOS_token]]).to(device)

    teacher_forcing = True if random.random() <= model_args.teacher_forcing_ratio else False

    for idx_dec in range(len(y_tensor[0])): # We only have one Batch at this moment
        if model_args.attention_decoder: # With attention
            dec_cell_log_prob_output, dec_selected_cls_idx, dec_selected_cls_log_prob, dec_cell_hidden, context_vec, attn_score = _helper_decode_attn(
                                                                        dec_input, dec_hidden,
                                                                        model_output_emb, model_dec,
                                                                        context_vec,
                                                                        enc_output, beam_search_length)
            dec_input = dec_selected_cls_idx
            dec_hidden = dec_cell_hidden
            dec_output = dec_cell_log_prob_output
        else: # w/o attention
            dec_output, dec_input, dec_input_val, dec_hidden = _helper_decode(dec_input, dec_hidden, model_output_emb,
                                                                              model_dec,
                                                                              model_args.beam_search_length)
        cur_output_char_loss = criterion(dec_output, y_tensor[:, idx_dec].type(torch.LongTensor).to(device))
        loss += cur_output_char_loss
        if dec_input == EOS_token:
            break
        if teacher_forcing:
            dec_input = y_tensor[0, idx_dec].unsqueeze(0).unsqueeze(0) # B(=1) X 1 v

    loss.backward()
    input_emb_optimizer.step()
    output_emb_optimizer.step()
    enc_optimizer.step()
    dec_optimizer.step()

    return loss/len(y_tensor[0])# normalizing the loss

def train_model_encdec(train_data: List[Example], dev_data: List[Example], input_indexer, output_indexer,
                       args) -> Seq2SeqSemanticParser:
    """
    Function to train the encoder-decoder model on the given data.
    :param train_data:
    :param dev_data: Development set in case you wish to evaluate during training
    :param input_indexer: Indexer of input symbols
    :param output_indexer: Indexer of output symbols
    :param args:
    :return:
    """
    parser = argparse.ArgumentParser(description='models.py')
    model_args = add_models_args(parser)

    if not model_args.batchfirst:
        raise Exception  # all implementation here is for Batch-first orders in all tensors

    if not model_args.n_layers:
        raise Exception  # The LSTM, specially the bidirectioanl, does not  take the n_layer > 1 into account at the moment

    if model_args.embed_size < 100 or model_args.hidden_size < 50:
        print("Embed size less than 100 and hidden size less than 50 usually do not end up with a good model.")

    # Preparing dataset batches
    train_dataset = CoderDecoderDataset(train_data, input_indexer, output_indexer, reverse_input=False)
    train_dataset_batches = DataLoader(dataset=train_dataset, batch_size=model_args.batch_size, shuffle=True)

    if args.print_dataset:
        print("Train length: %i" % train_dataset.x_max_sentence_len)
        print("Train output length: %i" % train_dataset.y_max_sentence_le)
        print("Train matrix: %s; shape = %s" % (train_dataset.x_sentences, train_dataset.x_sentences.shape))

    # Create encoder, decoder and embedding layers
    model_input_emb = EmbeddingLayer(model_args.embed_size, len(input_indexer), model_args.embedding_dropout).to(device)
    model_enc = RNNEncoder(model_args).to(device)
    model_output_emb = EmbeddingLayer(model_args.embed_size, len(output_indexer), model_args.embedding_dropout).to(device)

    if not model_args.attention_decoder:
        print("Decoder model with no attention")
        model_dec = RNNDecoder(len(output_indexer), model_args).to(device)
    else:
        print("Decoder model with attention")
        model_dec = LuongAttnDecoderRNN(num_output_classes=len(output_indexer), model_args=model_args).to(device)

    # Initialize optimizers
    input_emb_optimizer = optim.Adam(model_input_emb.parameters(), lr=args.lr)
    enc_optimizer = optim.Adam(model_enc.parameters(), lr=args.lr)
    output_emb_optimizer = optim.Adam(model_output_emb.parameters(), lr=args.lr)
    dec_optimizer = optim.Adam(model_dec.parameters(), lr=args.lr)

    loss_per_epoch = []
    print("Start training for {} epochs ...".format(model_args.epochs))
    # setting them all in train mode
    for epoch in range(model_args.epochs):
        total_loss = 0
        # loop over various batches
        for i, (x_sens, x_sens_lens, _, y_sens, y_sens_lens, _) in enumerate(train_dataset_batches):
            x_sens, x_sens_lens, y_sens, y_sens_lens = x_sens.to(device), x_sens_lens.to(device), y_sens.to(
                device), y_sens_lens.to(device)

            if len(x_sens_lens) == model_args.batch_size:
                loss = train_for_a_batch_of_data(x_sens, x_sens_lens, y_sens, y_sens_lens,
                                          model_input_emb, model_enc, model_output_emb, model_dec,
                                          enc_optimizer, dec_optimizer, input_emb_optimizer, output_emb_optimizer,
                                          input_indexer, output_indexer,
                                          model_args)

                total_loss += loss.item()
        loss_per_epoch.append(total_loss)
        print("The loss at epoch {} is {}.".format(epoch, total_loss))
        # Plotting the loss vs epoch
        # if epoch % model_args.loss_plot_frequency == 0:
        #     tensor_board_logger.add_scalar("avg_loss", total_loss, global_step=epoch)
        if epoch % model_args.model_evaluation_frequency == 0:
            seq_2_seq_decoder_model = Seq2SeqSemanticParser(model_input_emb, model_enc, model_output_emb,
                                                                        model_dec, model_args, output_indexer)
            selected_test_data = random.choice(dev_data)
            res = evaluate([selected_test_data], seq_2_seq_decoder_model, example_freq=50, print_output=True,
                           outfile=None, use_java=False)
            # Just for plotting the attention
            _ = seq_2_seq_decoder_model.decode([selected_test_data], attention_plot=True)
            print("The result for a random devtest dataset is: {}".format(res))

    seq_2_seq_decoder_model = Seq2SeqSemanticParser(model_input_emb, model_enc, model_output_emb, model_dec,
                                                                model_args, output_indexer)
    res = evaluate(dev_data, seq_2_seq_decoder_model, example_freq=50, print_output=True, outfile=None, use_java=False)
    return seq_2_seq_decoder_model
