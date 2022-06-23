from recommenders.KNNRecommender import KNNRecommender
import random
import numpy as np


def random_search(rec, k):
    params_knn = {'model': ['sknn', 's-sknn', 'sf-sknn'],
                  'k': list(np.arange(5, 30))}
    params_gru4rec = {'session_layers': [list(np.arange(5, 30))],
                      'batch_size': list(np.arange(5, 50)),
                      'learning_rate': list(np.arange(0, 1, 0.05)),
                      'momentum': list(np.arange(0.1, 0.5, 0.05)),
                      'dropout': list(np.arange(0, .5, 0.05)),
                      'epochs': list(np.arange(5, 15))}
    if rec == 'knn':
        recs = []
        for i in range(k):
            params = {i[0]: random.choice(i[1]) for i in params_knn.items()}
            rec_sknn = KNNRecommender(**params)
            recs.append(rec_sknn)
        return recs
    elif rec == 'sasrec':
        pass
