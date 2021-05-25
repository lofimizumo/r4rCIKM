from recommenders.AttentionRecommender import AttentionRecommender
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.datasets.amazon import get_amazon_dataset
from spotlight.cross_validation import random_train_test_split

import numpy as np
from recommenders.Rec4RecRecommender import Rec4RecRecommender
from recommenders.Rec4RecMk2 import R4RRecommender
from recommenders.KNNRecommender import KNNRecommender
from recommenders.RNNRecommender import RNNRecommender
from recommenders.FPMCRecommender import FPMCRecommender
from recommenders.PopularityRecommender import PopularityRecommender
import torch
from util.attention.utils import *
from util import evaluation
from util.model_factory import random_search
from util.make_data import *
from util.metrics import mrr, recall
import os
import argparse
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def drawRecEmbeddings(rec_names, data):

    # dt = ensemble.model.seq_embs.detach().cpu().numpy()
    # data = ensemble.model.rec_emb.weight[:len(
    #     rec_ensemble)].detach().cpu().numpy()

    # dt = np.concatenate([dt, data], 0)

    m = TSNE(learning_rate=80, perplexity=1, n_iter=500)
    xy = m.fit_transform(data)

    df = pd.DataFrame(data)
    df['x'] = xy[:, 0]
    df['y'] = xy[:, 1]
    df['rec'] = 'item'
    df['rec'] = rec_names
    sns.set_style("whitegrid")
    sns.scatterplot(x='x', y='y', hue='rec', style='rec', s=100,
                    data=df, markers=True)
    plt.show()


def make_tr_te(dataset):
    """
    dataset = {'ml','lastfm','amazon'}
    """
    if dataset == 'ml':
        ml_100K = get_movielens_dataset()
        tr, te = random_train_test_split(ml_100K)
        tr = spotlight_to_pandas(ml_100K)
        te = spotlight_to_pandas(te)
        valid_sequences = te.loc[te['sequence'].map(
            len) > abs(1), 'sequence'].values[:100]
        te_array = te.loc[te['sequence'].map(
            len) > abs(1), 'sequence'].values[100:]

    if dataset == 'lastfm':
        tr, te = make_data_toy_data()
        # te, u = get_test_sequences_and_users(te, 1, tr['user_id'].values)
        valid_sequences = te.loc[te['sequence'].map(
            len) > abs(1), 'sequence'].values[:500]
        te_array = te.loc[te['sequence'].map(
            len) > abs(1), 'sequence'].values[500:600]

    if dataset == 'amazon':
        amazon = get_amazon_dataset(
            min_user_interactions=10, min_item_interactions=10)
        _, amazon = random_train_test_split(amazon, 0.01)
        tr, te = random_train_test_split(amazon)
        tr = spotlight_to_pandas(tr)
        te = spotlight_to_pandas(te)
        te = filter_testset(tr, te)
        valid_sequences = te.loc[te['sequence'].map(
            len) > abs(1), 'sequence'].values[:100]
        te_array = te.loc[te['sequence'].map(
            len) > abs(1), 'sequence'].values[100:]

    return tr, te_array, valid_sequences


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=100)
    parser.add_argument('--n_iter', type=int, default=20)
    parser.add_argument('--support', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=.5e-2)
    parser.add_argument('--l2', type=float, default=3e-3)
    parser.add_argument('--neg_samples', type=int, default=3)
    parser.add_argument('--maxlen', type=int, default=500)
    parser.add_argument('--hidden_units', type=int, default=100)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--sets_of_neg_samples', type=int, default=50)
    config = parser.parse_args()
    METRICS = {'recall': recall}

    tr, te, valid_sequences = make_tr_te('lastfm')
    # rec_sasrec = AttentionRecommender()
    # rec_sasrec.fit(tr)
    # eval_score = evaluation.sequential_evaluation(
    #     rec_sasrec, te, METRICS.values(), None, 1, 1, 20, scroll=False)
    # print(eval_score)

    rec_knns = random_search('knn', 30)
    rec_gru4recs = random_search('gru4rec', 3)

    rec_ensemble = rec_knns

    for rec in rec_ensemble:
        rec.fit(tr)

    ensemble = R4RRecommender(
        item_count, 100, rec_ensemble, config, pretrained_embeddings=None)
    # ensemble.fit(valid_sequences, METRICS, input_space='f')
    ensemble.fit(valid_sequences, METRICS, input_space='f', users=None)

    ensemble_eval_score = evaluation.sequential_evaluation(
        ensemble, test_sequences=te, evaluation_functions=METRICS.values(), top_n=10, scroll=False)
    print(ensemble_eval_score)
    fitnesses = torch.stack(ensemble.rec_fitnesses).T.detach().numpy()
    pd.DataFrame(fitnesses).to_csv(
        'fitness.csv', mode='a', index=False, header=False)
    drawRecEmbeddings(
        [str(i) for i in rec_ensemble], fitnesses)
