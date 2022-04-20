#%%
print('a')

#%%



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
    METRICS = {'recall': recall}
    train_data, test_data = make_toy_data()
    sequences = get_test_sequences(train_data, 1)
    test_sequences = get_test_sequences(test_data, 1)
    item_count = item_count(train_data, 'sequence')

    rec_srgnn = SRGNNRecommender()
    rec_mf = Prod2VecRecommender(
        size=100, min_count=2, window=5, decay_alpha=0.9, workers=4)
    rec_mf2 = Prod2VecRecommender(size=50, window=3)
    rec_sknn = KNNRecommender(model='sknn', k=12)
    rec_sknn2 = KNNRecommender(model='sknn', k=10)
    rec_sknn3 = KNNRecommender(model='sknn', k=8)
    rec_sknn4 = KNNRecommender(model='sknn', k=6)
    rec_sknn5 = KNNRecommender(model='sknn', k=4)
    rec_gru4rec = RNNRecommender(session_layers=[
                                 20], batch_size=16, learning_rate=0.1, momentum=0.1, dropout=0.1, epochs=5)
    rec_gru4rec2 = RNNRecommender(session_layers=[
        10], batch_size=8, learning_rate=0.1, momentum=0.05, dropout=0, epochs=5)
    # rec_ensemble = [rec_sknn, rec_sknn2, rec_mf2,
    #                 rec_mf, rec_gru4rec, rec_gru4rec2]
    rec_ensemble = [rec_srgnn]
    for rec in rec_ensemble:
        rec.fit(train_data)

    rec_eval_scores = evaluation.predict_score_of_sequences(rec_ensemble,
                                                            test_sequences=sequences,
                                                            given_k=1,
                                                            look_ahead=1,
                                                            evaluation_functions=METRICS.values(),
                                                            top_n=10,
                                                            scroll=False  # notice that scrolling is disabled!
                                                            )
    np_sub_scores = np.array(rec_eval_scores)
    sub_scores = np_sub_scores.sum(0)/np_sub_scores.shape[0]
    print(sub_scores)
    seq, labels_and_negs = label(test_sequences, rec_eval_scores, 5)
    lst = list(seq)
    lst = [list(map(int, i)) for i in lst]
    seq = np.asarray(lst)

    # ensemble = Rec4RecRecommender(
    #     item_count, 100, rec_ensemble, config, pretrained_embeddings=None)
    # ensemble.fit(seq, labels_and_negs)

    # ensemble_eval_score = evaluation.sequential_evaluation(
    #     ensemble, test_sequences=test_sequences, evaluation_functions=METRICS.values(), top_n=10, scroll=False)
    # print(ensemble_eval_score, '\n', sub_scores)

    # draw the embedding distance using TSNE

    def drawItemSequences(data, rec_hue=None):
        dt = data
        from sklearn.manifold import TSNE
        m = TSNE(learning_rate=50)
        xy = m.fit_transform(dt)
        import seaborn as sns
        import matplotlib.pyplot as plt
        df = pd.DataFrame(data=dt, columns=[
                          'sknn', 'mf', 'gru4rec'])
        df['x'] = xy[:, 0]
        df['y'] = xy[:, 1]
        sns.scatterplot(x='x', y='y', hue=rec_hue, data=df)
        plt.show()

    # drawItemSequences(rec_eval_scores, 'sknn')
    # drawItemSequences(rec_eval_scores, 'sknn-2')
    # drawItemSequences(rec_eval_scores, 'mf')
    # drawItemSequences(rec_eval_scores, 'mf2')
    # drawItemSequences(rec_eval_scores, 'gru4rec')
    def drawRecEmbeddings(rec_names):
        import torch
        dt = ensemble.model.seq_embs.detach().cpu().numpy()
        rec_embs = ensemble.model.rec_emb.weight[:len(
            rec_ensemble)].detach().cpu().numpy()
        dt = np.concatenate([dt, rec_embs], 0)
        column_rec = np.zeros(dt.shape[0])
        # column_rec[-len(rec_ensemble):] = np.arange(1, len(rec_ensemble)+1)
        # column_rec[-len(rec_ensemble):] = np.array(rec_names)
        # column_rec = np.expand_dims(column_rec, 1)
        # dt = np.concatenate([dt, column_rec], 1)
        from sklearn.manifold import TSNE
        m = TSNE(learning_rate=80)
        xy = m.fit_transform(dt)
        import seaborn as sns
        import matplotlib.pyplot as plt

        df = pd.DataFrame(dt)
        df['x'] = xy[:, 0]
        df['y'] = xy[:, 1]
        df['rec'] = 'item'
        df['rec'][-len(rec_ensemble):] = rec_names
        df['style'] = df['rec']
        df['size'] = 1
        df['size'][-len(rec_ensemble):] = 4
        markers = {'item': 'v', 'rec_sknn': 's', 'rec_sknn_2': 'o',
                   'rec_mf': '^', 'rec_mf2': 'p', 'rec_gru4rec': '*'}
        sns.scatterplot(x='x', y='y', size='size', hue='rec',
                        data=df, markers=['o', '*', '+', '^', 's', 'p', 'v'])
        plt.show()

    # drawRecEmbeddings(
    #     ['rec_'+str(i) for i in range(6)])

    def drawEmbeddings(data):
        dt = ensemble.model.rec_emb.weight[:len(
            rec_ensemble)].detach().cpu().numpy()
        from sklearn.manifold import TSNE
        m = TSNE(learning_rate=50)
        xy = m.fit_transform(dt)
        import seaborn as sns
        import matplotlib.pyplot as plt
        df = pd.DataFrame(data=dt)
        df['x'] = xy[:, 0]
        df['y'] = xy[:, 1]
        sns.scatterplot(x='x', y='y', data=df)
        plt.show()
