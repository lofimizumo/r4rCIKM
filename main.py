from recommenders.AttentionRecommender import AttentionRecommender
import numpy as np
from recommenders.Rec4RecRecommender import Rec4RecRecommender
from recommenders.Rec4RecMk2 import R4RRecommender
from recommenders.KNNRecommender import KNNRecommender
from recommenders.RNNRecommender import RNNRecommender
from util.attention.utils import pandas_data_to_SASRec, data_partition
from util import evaluation
from util.make_data import *
from util.metrics import mrr, recall
import os
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=100)
    parser.add_argument('--n_iter', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=.5e-2)
    parser.add_argument('--l2', type=float, default=3e-3)
    parser.add_argument('--neg_samples', type=int, default=3)
    parser.add_argument('--maxlen', type=int, default=100)
    parser.add_argument('--hidden_units', type=int, default=100)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--sets_of_neg_samples', type=int, default=50)
    config = parser.parse_args()
    METRICS = {'recall': recall}
    sequences, test_sequences = make_data_toy_data()
    valid_sequences = test_sequences.loc[test_sequences['sequence'].map(
        len) > abs(1), 'sequence'].values[:100]
    test_sequences = test_sequences.loc[test_sequences['sequence'].map(
        len) > abs(1), 'sequence'].values[100:600]
    item_count = item_count(sequences, 'sequence')

    # rec_sasrec = AttentionRecommender()
    # data_sasrecformat = pandas_data_to_SASRec(sequences, 'sequence', 'user_id')
    # train_data_sasrec = data_partition(data_sasrecformat)
    # rec_sasrec.fit(train_data_sasrec)
    # eval_score = evaluation.sequential_evaluation(
    #     rec_sasrec, test_sequences, METRICS.values(), None, 1, 1, 10, scroll=False)
    # print(eval_score)

    rec_sknn2 = KNNRecommender(model='sknn', k=10)
    rec_sknn = KNNRecommender(model='sknn', k=12)
    # rec_sknn.fit(sequences)
    # eval_score = evaluation.sequential_evaluation(
    #     rec_sknn, test_sequences, METRICS.values(), None, 1, 1, 10, scroll=False)
    # print(eval_score)
    rec_gru4rec = RNNRecommender(session_layers=[
                                 20], batch_size=16, learning_rate=0.2, momentum=0.1, dropout=0.1, epochs=10)
    rec_ensemble = [rec_sknn, rec_gru4rec]
    for rec in rec_ensemble:
        rec.fit(sequences)

    ensemble = Rec4RecRecommender(
        item_count, 100, rec_ensemble, config, pretrained_embeddings=None)
    ensemble.fit(valid_sequences, METRICS)

    ensemble_eval_score = evaluation.sequential_evaluation(
        ensemble, test_sequences=test_sequences, evaluation_functions=METRICS.values(), top_n=10, scroll=False)
    print(ensemble_eval_score)
