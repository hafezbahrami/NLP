import argparse
import random
import numpy as np
from lf_evaluator import *
from models import *
from data import *
from utils import *
from typing import List


def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='main.py')

    # General system running and configuration options
    parser.add_argument('--do_nearest_neighbor', dest='do_nearest_neighbor', default=False, action='store_true',
                        help='run the nearest neighbor model')
    parser.add_argument('--task', type=str, default='parsing', help='Task here can be parsing or translation')

    parser.add_argument('--train_path', type=str, default='data/geo_train.tsv', help='path to train data')
    parser.add_argument('--dev_path', type=str, default='data/geo_dev.tsv', help='path to dev data')
    parser.add_argument('--test_path', type=str, default='data/geo_test.tsv', help='path to blind test data')
    parser.add_argument('--test_output_path', type=str, default='geo_test_output.tsv',
                        help='path to write blind test results')
    parser.add_argument('--domain', type=str, default='geo', help='domain (geo for geoquery)')
    parser.add_argument('--no_java_eval', dest='perform_java_eval', default=False, action='store_false',
                        help='run evaluation of constructed query against java backend')
    parser.add_argument('--print_dataset', dest='print_dataset', default=False, action='store_true',
                        help="Print some sample data on loading")
    parser.add_argument('--eval_from_checkpoint', default=False, action='store_true',
                        help="Evaluate model from checkpoint")
    parser.add_argument('--model_path', type=str, default='final_model.pt', help='path to model checkpoint')

    parser.add_argument('--train_path_for_translation', type=str, default='data/eng-fra.txt', help='path to train data')
    parser.add_argument('--min_sent_len', type=int, default=10, help='Min sentence length to keep it in the training for translation.')
    parser.add_argument('--max_sent_len', type=int, default=20, help='Max sentence length to keep it in the training for translation.')
    parser.add_argument('--reverse', default=False, action='store_true', help="Reverse the sentence pair order in the training dataset.")


    add_models_args(parser)  # defined in models.py

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Load the training and test data

    if args.task == "parsing":
        print("Loading parsing dataset: ...")
        train, dev, test = load_datasets(args.train_path, args.dev_path, args.test_path, domain=args.domain,
                                         task=args.task, min_sent_len=args.min_sent_len, max_sent_len=args.max_sent_len,
                                         reverse=args.reverse)
    else:
        print("Loading translation dataset: ...")
        train, dev, test = load_datasets(args.train_path_for_translation, args.train_path_for_translation,
                                         args.train_path_for_translation, domain=args.domain, task=args.task,
                                         min_sent_len=args.min_sent_len, max_sent_len=args.max_sent_len,
                                         reverse=args.reverse)
    train_data_indexed, dev_data_indexed, test_data_indexed, input_indexer, output_indexer = index_datasets(train, dev,
                                                                                                            test,
                                                                                                            args.decoder_len_limit)

    print("%i train exs, %i dev exs, %i input types, %i output types" % (
    len(train_data_indexed), len(dev_data_indexed), len(input_indexer), len(output_indexer)))
    if args.print_dataset:
        print("Input indexer: %s" % input_indexer)
        print("Output indexer: %s" % output_indexer)
        print("Here are some examples post tokenization and indexing:")
        for i in range(0, min(len(train_data_indexed), 10)):
            print(train_data_indexed[i])
    if not args.eval_from_checkpoint:
        if args.do_nearest_neighbor:
            decoder = NearestNeighborSemanticParser(train_data_indexed)
        else:
            decoder = train_model_encdec(train_data_indexed, dev_data_indexed, input_indexer, output_indexer,
                                         args)
            torch.save(decoder, args.model_path)
    else:
        decoder = torch.load(args.model_path)
    print("=======DEV SET=======")
    evaluate(dev_data_indexed, decoder, use_java=args.perform_java_eval)
    print("=======FINAL PRINTING ON BLIND TEST=======")
    evaluate(test_data_indexed, decoder, print_output=False, outfile="geo_test_output.tsv",
             use_java=args.perform_java_eval)
