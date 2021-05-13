from util.split import last_session_out_split, random_holdout
from util.data_utils import *
import datetime
import numpy
import sys
import os
from random import randint
import pathlib
sys.path.append(os.getcwd())


def make_data_toy_data():
    """
    return: train_data, test_data
    """
    dataset_path = 'datasets/sessions_sample_10.csv'
    dataset = create_seq_db_filter_top_k(
        path=dataset_path, topk=1000, last_months=1)
    from collections import Counter
    cnt = Counter()
    dataset.sequence.map(cnt.update)
    train_data, test_data = last_session_out_split(dataset)
    return train_data, test_data


def unique_item_count(sequences, sequence_column_name):
    """
    input:Dataframe sequences
    """
    df = sequences.explode(sequence_column_name)
    unique_items = df[sequence_column_name].unique()
    return unique_items


def item_count(sequences, sequence_column_name):
    """
    input:Dataframe sequences
    """
    item_max_id = sequences[sequence_column_name].map(max).max()
    return int(item_max_id)


def label(train_data, rec_eval_scores, sequence_length):
    """
    generate one label and one negative sample:
    [(ground_truth,negative sample)]
    example:
    [0,1,0,1,0]
    [1,0,1,0,0]
    out:
    (2,3)
    (1,2)
    """
    seq_length = sequence_length
    rows_to_discard = []
    labels_and_negs = []
    for row_index, row in enumerate(rec_eval_scores):
        positive_index = []
        negative_index = []
        for i, score in enumerate(row):
            if score == 1:
                positive_index.append(i)
            else:
                negative_index.append(i)
        if negative_index == [] or positive_index == []:
            rows_to_discard.append(row_index)
            continue
        labels_and_negs.append((positive_index[randint(
            0, len(positive_index)-1)], negative_index[randint(0, len(negative_index)-1)]))
    ret = numpy.delete(train_data, rows_to_discard, axis=0)
    labels_and_negs = numpy.asarray(labels_and_negs)
    return ret, labels_and_negs
