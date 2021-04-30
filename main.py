import numpy as np
from recommenders.Rec4RecRecommender import Rec4RecRecommender
from recommenders.KNNRecommender import KNNRecommender
from recommenders.RNNRecommender import RNNRecommender

from util import evaluation
from util.make_data import * 
from util.metrics import mrr,recall
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
    item_count = item_count(sequences, 'sequence')

    rec_sknn = KNNRecommender(model='sknn', k=12)
    rec_gru4rec = RNNRecommender(session_layers=[
                                 20], batch_size=16, learning_rate=0.1, momentum=0.1, dropout=0.1, epochs=5)
    rec_ensemble = [rec_sknn, rec_gru4rec]
    for rec in rec_ensemble:
        rec.fit(sequences)

    ensemble = Rec4RecRecommender(
        item_count, 100, rec_ensemble, config, pretrained_embeddings=None)
    ensemble.fit(test_sequences,METRICS)

    ensemble_eval_score = evaluation.sequential_evaluation(
        ensemble, test_sequences=test_sequences, evaluation_functions=METRICS.values(), top_n=10, scroll=False)

    