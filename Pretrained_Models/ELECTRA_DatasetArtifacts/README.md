# Dataset Artifacts: using ELECTRA-small pretrained model


## Project Scope: Exploring the Dataset Artifacts
Pre-trained models can often achieve high performance on benchmark datasets, but are they really “solving”
the tasks these datasets encapsulate? Sometimes a model can work extremely well even when presented with
a modified version of the input where it should not be possible to predict the right answer, like hypothesisonly
baselines in NLI (Poliak et al., 2018), which calls into question what the model is even learning.
Sometimes it is possible to find or construct examples very similar to those in the training data where the
model achieves surprisingly low performance. These include “contrast examples” (Gardner et al., 2020)
which are produced by modifying real examples in a small way, as well as adversarial examples (Jia and
Liang, 2017) and checklist examples (Ribeiro et al., 2020).

These observations all stem from the fact that a model may achieve high performance on a dataset by
learning spurious correlations, also called **dataset artifacts**. _The model is then expected to fail in settings 
where these artifacts are not present, which may include real-world testbeds of interest_.

The goal here is to investigate the performance of a pre-trained model on a task of "Quasetion and Answer", using SQuAD dataset.

## Analysis:
I started by training a model on the dataset and doing some analysis of it, using the starter code. I used
the ELECTRA-small (Clark et al., 2020) model; ELECTRA has the same architecture as BERT with an improved 
training method, and the small model is computationally easier to run than larger models. 

There are many ways to investigate the artifacts in the datasets:
	1 (changing data) Use contrast sets (Gardner et al., 2020), either ones that have already been constructed
or a small set of examples that you hand-design and annotate
	2 (changing data) Use checklist sets (Ribeiro et al., 2020)
	3 (changing data) Use adversarial challenge sets (Jia and Liang, 2017; Wallace et al., 2019; Bartolo et
al., 2020; Glockner et al., 2018; McCoy et al., 2019)
	4 (changing model) Use model ablations (hypothesis-only NLI, a sentence-factored model for multihop
question answering, a question/passage only model for QA) (Poliak et al., 2018; Chen and Durrett,
2019; Kaushik and Lipton, 2018)
	5 (statistical test) Use the “competency problems” framework: find spurious n-gram correlations with
answers (Gardner et al., 2021)

## Starter Code
**run.py**, a script which implements basic model training and evaluation using the HuggingFace transformers library.
 For information on the arguments to run.py and hints on how to extend its behavior, see the comments in the source and the repository’s
README at: https://github.com/gregdurrett/fp-dataset-artifacts.

HuggingFace The skeleton code is heavily based on HuggingFace transformers, which is an opensource
library providing implementations of pre-trained deep learning models for a variety of (mainly NLP)
tasks. If you want to get more familiar with transformers, you can check out the examples in their
GitHub repository.
Computational resources Even with the ELECTRA-small model, training with CPU only on a large
dataset like SNLI or SQuAD can be time-consuming. Please see the section on compute and feasibility at
the end of the project spec for some options.




## Getting Started
You'll need Python >= 3.6 to run the code in this repo.

First, clone the repository:

`git clone git@github.com:gregdurrett/fp-dataset-artifacts.git`

Then install the dependencies:

`pip install --upgrade pip`

`pip install -r requirements.txt`

If you're running on a shared machine and don't have the privileges to install Python packages globally,
or if you just don't want to install these packages permanently, take a look at the "Virtual environments"
section further down in the README.

To make sure pip is installing packages for the right Python version, run `pip --version`
and check that the path it reports is for the right Python interpreter.


## Training and evaluating a model
To train an ELECTRA-small model on the SNLI natural language inference dataset, you can run the following command:

`python3 run.py --do_train --task nli --dataset snli --output_dir ./trained_model/`

Checkpoints will be written to sub-folders of the `trained_model` output directory.
To evaluate the final trained model on the SNLI dev set, you can use

`python3 run.py --do_eval --task nli --dataset snli --model ./trained_model/ --output_dir ./eval_output/`

To prevent `run.py` from trying to use a GPU for training, pass the argument `--no_cuda`.

To train/evaluate a question answering model on SQuAD instead, change `--task nli` and `--dataset snli` to `--task qa` and `--dataset squad`.

**Descriptions of other important arguments are available in the comments in `run.py`.**

Data and models will be automatically downloaded and cached in `~/.cache/huggingface/`.
To change the caching directory, you can modify the shell environment variable `HF_HOME` or `TRANSFORMERS_CACHE`.
For more details, see [this doc](https://huggingface.co/transformers/v4.0.1/installation.html#caching-models).

An ELECTRA-small based NLI model trained on SNLI for 3 epochs (e.g. with the command above) should achieve an accuracy of around 89%, depending on batch size.
An ELECTRA-small based QA model trained on SQuAD for 3 epochs should achieve around 78 exact match score and 86 F1 score.

## Working with datasets
This repo uses [Huggingface Datasets](https://huggingface.co/docs/datasets/) to load data.
The Dataset objects loaded by this module can be filtered and updated easily using the `Dataset.filter` and `Dataset.map` methods.
For more information on working with datasets loaded as HF Dataset objects, see [this page](https://huggingface.co/docs/datasets/process.html).

## Virtual environments
Python 3 supports virtual environments with the `venv` module. These will let you select a particular Python interpreter
to be the default (so that you can run it with `python`) and install libraries only for a particular project.
To set up a virtual environment, use the following command:

`python3 -m venv path/to/my_venv_dir`

This will set up a virtual environment in the target directory.
WARNING: This command overwrites the target directory, so choose a path that doesn't exist yet!

To activate your virtual environment (so that `python` redirects to the right version, and your virtual environment packages are active),
use this command:

`source my_venv_dir/bin/activate`

This command looks slightly different if you're not using `bash` on Linux. The [venv docs](https://docs.python.org/3/library/venv.html) have a list of alternate commands for different systems.

Once you've activated your virtual environment, you can use `pip` to install packages the way you normally would, but the installed
packages will stay in the virtual environment instead of your global Python installation. Only the virtual environment's Python
executable will be able to see these packages.
