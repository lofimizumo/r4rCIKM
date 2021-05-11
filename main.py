from recommenders.AttentionRecommender import AttentionRecommender
import numpy as np
from recommenders.Rec4RecRecommender import Rec4RecRecommender
from recommenders.KNNRecommender import KNNRecommender
from recommenders.RNNRecommender import RNNRecommender
from util.attention.utils import pandas_data_to_SASRec, data_partition
from util import evaluation
from util.make_data import *
from util.metrics import mrr, recall
import os
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_test_sequences(test_data, given_k):
    # we can run evaluation only over sequences longer than abs(LAST_K)
    test_sequences = test_data.loc[test_data['sequence'].map(
        len) > abs(given_k), 'sequence'].values
    return test_sequences


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=100)
    parser.add_argument('--n_iter', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=.5e-2)
    parser.add_argument('--l2', type=float, default=3e-3)
    parser.add_argument('--neg_samples', type=int, default=3)
    parser.add_argument('--sets_of_neg_samples', type=int, default=50)
    config = parser.parse_args()
    METRICS = {'mrr': mrr}
    sequences, test_sequences = make_data_toy_data()
    test_sequences = test_sequences.loc[test_sequences['sequence'].map(
        len) > abs(1), 'sequence'].values
    item_count = item_count(sequences, 'sequence')

    data_sasrecformat = pandas_data_to_SASRec(sequences, 'sequence', 'user_id')
    train_data_sasrec = data_partition(data_sasrecformat)
    a=train_data_sasrec[0]

    rec_sasrec = AttentionRecommender()
    rec_sasrec.fit(train_data_sasrec)
    rec_sknn2 = KNNRecommender(model='sknn', k=10)
    rec_sknn = KNNRecommender(model='sknn', k=12)
    rec_gru4rec = RNNRecommender(session_layers=[
                                 10], batch_size=16, learning_rate=0.1, momentum=0.1, dropout=0.1, epochs=5)
    rec_ensemble = [rec_sknn, rec_sknn2]
    for rec in rec_ensemble:
        rec.fit(sequences)
    eval_score = evaluation.sequential_evaluation(
        rec_sknn, test_sequences, METRICS.values(), None, 1, 1, 10, scroll=False)
    print(eval_score)

    # ensemble = Rec4RecRecommender(
    #     item_count, 100, rec_ensemble, config, pretrained_embeddings=None)
    # ensemble.fit(test_sequences, METRICS)

    # ensemble_eval_score = evaluation.sequential_evaluation(
    #     ensemble, test_sequences=test_sequences, evaluation_functions=METRICS.values(), top_n=10, scroll=False)
