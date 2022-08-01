import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

import matplotlib.pyplot as plt
import seaborn as sns


# For QA we use the BertForQuestionAnswering, which is pretrained for SQuAD dataset.
qa_model_large_bert = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

tokenizer = BertTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')

# trying an example
question = "How many parameters does BERT-large have?"
answer_text = "BERT-large is really big... it has 24-layers and an embedding size of 1,024, for a " \
              "total of 340M parameters! Altogether it is 1.34GB, so expect it to take a couple minutes " \
              "to download to your Colab instance."

# Apply the tokenizer to the input text, treating them as a text-pair.
input_ids = tokenizer.encode(question, answer_text)

print('The input has a total of {:} tokens.'.format(len(input_ids)))

# BERT only needs the token IDs, but for the purpose of inspecting the
# tokenizer's behavior, let's also get the token strings and display them.
tokens = tokenizer.convert_ids_to_tokens(input_ids)
for token, id in zip(tokens, input_ids):
    if id == tokenizer.sep_token_id: # for [SEP] token, add some space
        print('')
    print('{:<12} {:>6,}'.format(token, id))
    if id == tokenizer.sep_token_id:
        print('')

# -------------------------------------------------------------------------
# Segment embedding to distinguish the question from the passage

sep_index = input_ids.index(tokenizer.sep_token_id) # idx for the 1st instance of [SEP] token
num_seg_a = sep_index + 1 # The number of segment A tokens includes the [SEP] token istelf.
num_seg_b = len(input_ids) - num_seg_a # The remainder are segment B.
segment_ids = [0]*num_seg_a + [1]*num_seg_b # Construct the list of 0s and 1s.
assert len(segment_ids) == len(input_ids) # There should be a segment_id for every input token.

# -----------------------------------------------------------------------
# Run our example through the model.
outputs = qa_model_large_bert(torch.tensor([input_ids]), # The tokens representing our input text.
                             token_type_ids=torch.tensor([segment_ids]), # The segment IDs to differentiate question from answer_text
                             return_dict=True)

start_scores = outputs.start_logits
end_scores = outputs.end_logits

answer_start = torch.argmax(start_scores) # Find the tokens with the highest `start` scores.
answer_end = torch.argmax(end_scores) # Find the tokens with the highest `end` scores.
answer = ' '.join(tokens[answer_start:answer_end+1]) # Combine the tokens in the answer and print it out.
print('Answer: "' + answer + '"')

# handling subwords in the answer
answer = tokens[answer_start]
for i in range(answer_start + 1, answer_end + 1):
    if tokens[i][0:2] == '##':
        answer += tokens[i][2:]
    else:
        answer += ' ' + tokens[i]
print('Answer: "' + answer + '"')

# --------------------------------------------------------------------------
# Visualization


sns.set(style='darkgrid') # Use plot styling from seaborn.
plt.rcParams["figure.figsize"] = (16,8)

def _helper_bar_plot_scores(x_lables, y_val, given_title: str = "None"):
    ax = sns.barplot(x=x_lables, y=y_val, ci=None)  # Create a barplot
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")  # Turn the xlabels vertical.
    ax.grid(True)  # Turn on the vertical grid to help align words to scores.
    plt.title(given_title)
    plt.show()

# Pull the scores out of PyTorch Tensors and convert them to 1D numpy arrays.
s_scores = start_scores.detach().numpy().flatten()
e_scores = end_scores.detach().numpy().flatten()

# We'll use the tokens as the x-axis labels. In order to do that, they all need
# to be unique, so we'll add the token index to the end of each one.
token_labels = []
for (i, token) in enumerate(tokens):
    token_labels.append('{:} - {:>2}'.format(token, i))

_helper_bar_plot_scores(x_lables=token_labels, y_val=s_scores, given_title='Start Word Scores')
_helper_bar_plot_scores(x_lables=token_labels, y_val=e_scores, given_title='End Word Scores')

