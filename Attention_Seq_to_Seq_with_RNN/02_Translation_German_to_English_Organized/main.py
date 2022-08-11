from typing import List
import copy
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

from process_data import get_data_and_vocab, predict_sentence
from model import Encoder, Decoder, Attention, Seq2Seq

BATCH_SIZE = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

en_vocab, de_vocab, train_iter, valid_iter, test_iter = get_data_and_vocab(BATCH_SIZE=BATCH_SIZE)

INPUT_DIM = len(de_vocab)
OUTPUT_DIM = len(en_vocab)
# ENC_EMB_DIM = 256
# DEC_EMB_DIM = 256
# ENC_HID_DIM = 512
# DEC_HID_DIM = 512
# ATTN_DIM = 64
# ENC_DROPOUT = 0.5
# DEC_DROPOUT = 0.5

ENC_EMB_DIM = 32
DEC_EMB_DIM = 32
ENC_HID_DIM = 64
DEC_HID_DIM = 64
ATTN_DIM = 8
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
model = Seq2Seq(enc, dec, device).to(device)

def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

model.apply(init_weights)

optimizer = optim.Adam(model.parameters())

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

PAD_IDX = en_vocab['<pad>']

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
# ------------------------------------------------------------------------------------

def train(model: nn.Module,
          iterator: torch.utils.data.DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):
    model.train()
    epoch_loss = 0
    for _, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output, _ = model(src, trg)
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        break
    return epoch_loss / len(iterator)

def print_sample_predicted_sent(src, trg, pred_trg):
    """print some translated sentences"""
    srl_l = list(src[:, 0].detach().cpu().numpy())
    trg_l = list(trg[:, 0].detach().cpu().numpy())
    pred_trg_l = list(pred_trg[:, 0].detach().cpu().numpy())

    print("Original sentence in German:")
    print(predict_sentence(srl_l), "German")
    print("Target English sentence:")
    print(predict_sentence(trg_l, "English"))
    print("Predicted translated English:")
    print(predict_sentence(pred_trg_l, "English"))
    print("----------------------------------------------------------------------")


def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             criterion: nn.Module):
    model.eval()
    epoch_loss = 0
    freq = len(iterator)//5
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)
            output, predicted_class = model(src, trg, 0) #turn off teacher forcing
            if i%freq >=0:
                print_sample_predicted_sent(src, trg, predicted_class)
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()


    return epoch_loss / len(iterator)

def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
CLIP = 1
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss = train(model, train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iter, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

test_loss = evaluate(model, test_iter, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

