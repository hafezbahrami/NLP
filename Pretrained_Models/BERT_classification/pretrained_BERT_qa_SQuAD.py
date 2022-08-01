# To run this script in Google Colab, look at the commands at the end of this script# Ref: https://www.youtube.com/watch?v=ZIRmXkHp0-c&t=2s

import os
import requests
# import json

import torch
from torch.utils.data import DataLoader

from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import AdamW
from tqdm import tqdm

from bert_fine_tunning_helper import BertQAHelper, SquadDataset

# ---------------------------------------------------------------------------------------------------
# Tensoboard plots in Google Colab
import io
import PIL.Image
import torchvision
from torchvision.transforms import ToTensor
import matplotlib
from matplotlib import pyplot as plt
# ---------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# TensorBoard stuff
import torch.utils.tensorboard as tb

log_dir = "./log/"  # This log directory should be consistent with the one defined in GoogleColab in "%tensorboard --logdir . --port 6006"
tensor_board_logger = tb.SummaryWriter(log_dir, flush_secs=1)
loss_plot_frequency = 5
# -------------------------------------------------------------------------------------------------------------


load_from_saved_model: bool = False
saved_model_path = 'models/distilbert-custom'
batch_size = 20
n_c = 20  # number of data to cosider, since the all data takes time to tune

# -------------------------------------------------------
if not os.path.exists('./data/benchmarks/squad'):
    os.makedirs('./data/benchmarks/squad')
# download the SQuAD data
url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/'
res = requests.get(f'{url}train-v2.0.json')

for file in ['train-v2.0.json', 'dev-v2.0.json']:
    res = requests.get(f'{url}{file}')
    # write to file
    with open(f'./data/benchmarks/squad/{file}', 'wb') as f:
        for chunk in res.iter_content(chunk_size=4):
            f.write(chunk)

print("getting the BERT model and tokenizer ... ")
if not load_from_saved_model:
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
else:

    model = DistilBertForQuestionAnswering.from_pretrained(saved_model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(saved_model_path)

bart_qa_helper = BertQAHelper(train_dataset_path='./data/benchmarks/squad/train-v2.0.json',
                              test_dataset_path='./data/benchmarks/squad/dev-v2.0.json',
                              tokenizer=tokenizer)

train_contexts, train_questions, train_answers = bart_qa_helper.read_squad("train")
val_contexts, val_questions, val_answers = bart_qa_helper.read_squad("test")

train_contexts, train_questions, train_answers = train_contexts[:n_c], train_questions[:n_c], train_answers[:n_c]
val_contexts, val_questions, val_answers = val_contexts[:n_c], val_questions[:n_c], val_answers[:n_c]

# adding end_idx to each question
bart_qa_helper.add_end_idx(train_answers, train_contexts)
bart_qa_helper.add_end_idx(val_answers, val_contexts)

# ----------------------------------------------------------
# Encode

print("start encoding")

train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

# A couple of examples from methods avilable from tokenizer
first_context_sen_input_ids = train_encodings.data["input_ids"][0]
first_context_sen_tokens = tokenizer.convert_ids_to_tokens(
    first_context_sen_input_ids)  # or tokenizer.decode( first_context_sen_input_ids )
print("A sample sentence tokens extracted from input ids: ....")
print(" ".join(first_context_sen_tokens))
print("\ninput id for [SEP] token is: ", tokenizer.sep_token_id)  # also tokenizer.cls_token or tokenizer.cls_token_id
print("\nAs another example the input_ids for \"Hafez is cool\" is: ",
      tokenizer.convert_tokens_to_ids(["Hafez", "is", "cool"]))

# start and end idx are stored based on idx position in the str of original context. Convert it to the encoding
# position which only contains the input_ids of each word.
bart_qa_helper.add_token_positions(train_encodings, train_answers)
bart_qa_helper.add_token_positions(val_encodings, val_answers)

# ------------------------------------------------------------------
# Loading data and start training
print("loading the data")
train_dataset = SquadDataset(train_encodings)
val_dataset = SquadDataset(val_encodings)

params = list(model.named_parameters())
# for p in params:
# print("{} {}".format(p[0], str(tuple(p[1].size()))))

print("Done with downloading model")
# setup GPU/CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# move model over to detected device
model.to(device)
# activate training mode of model
model.train()
# initialize adam optimizer with weight decay (reduces chance of overfitting)
optim = AdamW(model.parameters(), lr=5e-5)

# initialize data loader for training data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_sampler = SequentialSampler(val_dataset)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
print("started training")
counter = 0
for epoch in range(3):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epoch))
    print('Training...')

    # set model to train mode
    model.train()
    # setup loop (we use tqdm for the progress bar)
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        counter += 1
    # initialize calculated gradients (from prev step)
    optim.zero_grad()
    # pull all the tensor batches required for training
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    start_positions = batch['start_positions'].to(device)
    end_positions = batch['end_positions'].to(device)
    # train model on batch and return outputs (incl. loss)
    outputs = model(input_ids, attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions)
    # extract loss
    loss = outputs[0]
    # calculate loss for every parameter that needs grad update
    loss.backward()
    # update parameters
    optim.step()
    # print relevant info to progress bar
    loop.set_description(f'Epoch {epoch}')
    loop.set_postfix(loss=loss.item())
    if counter % loss_plot_frequency == 0:
        tensor_board_logger.add_scalar("avg_loss", loss.item(), global_step=counter)

# save the fine-tuned model
model_path = saved_model_path
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
# switch model out of training mode
model.eval()

acc = []

# initialize loop for progress bar
loop = tqdm(val_loader)
# loop through batches
for batch in loop:
    # we don't need to calculate gradients as we're not training
    with torch.no_grad():
        # pull batched items from loader
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_true = batch['start_positions'].to(device)
        end_true = batch['end_positions'].to(device)
        # make predictions
        outputs = model(input_ids, attention_mask=attention_mask)
        # pull preds out
        start_pred = torch.argmax(outputs['start_logits'], dim=1)  # outputs['start_logits']: B X seq_len & start_pred: B
        end_pred = torch.argmax(outputs['end_logits'], dim=1)
        # calculate accuracy for both and append to accuracy list
        acc.append(((start_pred == start_true).sum() / len(start_pred)).item())
        acc.append(((end_pred == end_true).sum() / len(end_pred)).item())
# calculate average accuracy in total
acc = sum(acc) / len(acc)

print(" Evaluation on the last eval batch ...")


def _helper_form_answer(answer_start_idx, answer_end_idx, tokens):
    answer = tokens[answer_start_idx]
    if answer_start_idx > answer_end_idx:
        return "Wrong prediction (start_idx > end_idx)"
    for i in range(answer_start_idx + 1, answer_end_idx + 1):
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
    else:
        answer += ' ' + tokens[i]
    return answer


print("T/F\tstart\tend\n")
for i in range(len(start_true)):
    tokens_for_last_eval_batch = tokenizer.convert_ids_to_tokens(input_ids[i, :])
    print(f"true indexes\t{start_true[i]}\t{end_true[i]}\n"
          f"pred indexes\t{start_pred[i]}\t{end_pred[i]}\n")
    print("True Answer is: {} \n".format(
        _helper_form_answer(start_true[i].item(), end_true[i].item(), tokens_for_last_eval_batch)))
    print("Predicted Answer is: {} \n".format(
        _helper_form_answer(start_pred[i].item(), end_pred[i].item(), tokens_for_last_eval_batch)))
    print("----------------------------------------------------------------------")

aaa = 1

# To Run it in Google Colab
# import torch
# print(torch.cuda.is_available())
# !pip install transformers
# !pip install wget
# import os
# if not os.path.exists('./data/benchmarks/squad'):
# os.makedirs('./data/benchmarks/squad')
#
# import os
# import wget
# import requests
# import json
#
# # download the SQuAD data
# url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/'
# res = requests.get(f'{url}train-v2.0.json')
#
# if not os.path.exists('./data/benchmarks/squad/train-v2.0.json'):
# wget.download(url + "train-v2.0.json", './data/benchmarks/squad/train-v2.0.json')
#
# if not os.path.exists('./data/benchmarks/squad/dev-v2.0.json'):
# wget.download(url + "dev-v2.0.json", './data/benchmarks/squad/dev-v2.0.json')
#
# !python3
# pretrained_BERT_qa_SQuAD.py
