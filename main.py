from sqlalchemy import true
from recommenders.Rec4Rec_proto_net import R4RProtoRecommender 
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
    dataset = {'ml1m','lastfm','amazon'}
    """
    if dataset == 'ml1m':
        ml1m = get_movielens_dataset()

        tr = ml1m.iloc[:int(len(ml1m)/1)]
        te = ml1m.loc[ml1m['sequence'].map(
            len) > abs(1), 'sequence'].values[int(len(ml1m)/2):]
        valid_sequences = te[:100]
        te_array = te[100:]

    if dataset == 'lastfm':
        tr, te = make_data_toy_data()
        # te, u = get_test_sequences_and_users(te, 1, tr['user_id'].values)
        # valid_sequences = te.loc[te['sequence'].map(
        #     len) > abs(1), 'sequence'].values[:500]
        # te_array = te.loc[te['sequence'].map(
        #     len) > abs(1), 'sequence'].values[500:600]
        valid_sequences = te.loc[te['sequence'].map(
            len) > abs(1), 'sequence'].values[:]
        te_array = te.loc[te['sequence'].map(
            len) > abs(1), 'sequence'].values[:]

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
    METRICS = {'recall': recall, 'mrr': mrr}

    # tr, te, valid_sequences = make_tr_te('lastfm')
    tr, te, valid_sequences = make_tr_te('ml1m')
    # rec_sasrec = AttentionRecommender()
    # rec_sasrec.fit(tr)
    # eval_score = evaluation.sequential_evaluation(
    #     rec_sasrec, te, METRICS.values(), None, 1, 1, 20, scroll=False)
    # print(eval_score)

    rec_knns = random_search('knn', 30)

    rec_ensemble = rec_knns

    for rec in rec_ensemble:
        rec.fit(tr)

    ensemble = R4RProtoRecommender(
        item_count, 100, rec_ensemble, config, pretrained_embeddings=None)
    # ensemble.fit(valid_sequences, METRICS, input_space='f')
    valid_sequences = valid_sequences[:int(len(valid_sequences)/10)]
    ensemble.fit(valid_sequences, METRICS, input_space='f', users=None)

    ensemble_eval_score = evaluation.sequential_evaluation(
        ensemble, test_sequences=te, evaluation_functions=METRICS.values(), top_n=10, scroll=False)
    print(ensemble_eval_score)
    fitnesses = torch.stack(ensemble.rec_fitnesses).T.detach().numpy()
    pd.DataFrame(fitnesses).to_csv(
        'fitness.csv', mode='a', index=False, header=False)
    drawRecEmbeddings(
        [str(i) for i in rec_ensemble], fitnesses)
