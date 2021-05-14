# %%
import numpy as np

# %%


def generateBaseEmbeddings(sequences, scores, support_size):
    """
        input:
        ----------------
        scores:[[1,0,0,1],[0,0,0,1]]
        scores[0]: score of rec0 e.g.:[1,0,0,1], rec1:[0,0,0,1]
        ----------------
        return: {'rec_k': top_performed_sequences}
        """
    ind_best_performed = np.flip(np.argsort(
        scores, axis=0), axis=0).transpose()
    ind_best_performed=ind_best_performed[:,:support_size].tolist()
    ret = {}
    for i, indices_rec_k in enumerate(ind_best_performed):
        rec_k_topK = [sequences[index] for index in indices_rec_k]
        ret[i] = rec_k_topK
    return ret


sequences = [[1, 4, 5, 6], [1, 2, 3, 4], [3, 54, 3, 24], [
    34, 2, 1, 4, 5, 6, 7], [23, 5, 6, 7, 5, 3], [3, 4, 2, 3, 1, 1, 2]]
scores = [[0, 1, 1, 0], [0, 1, 0, 0],[1, 0, 1, 1], [1, 1, 0, 0],[0,0,1,1],[0,1,1,0]]
np_scores = np.asarray(scores)
generateBaseEmbeddings(sequences, np_scores,3)

# %%


from spotlight.cross_validation import random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import rmse_score
from spotlight.factorization.explicit import ExplicitFactorizationModel

dataset = get_movielens_dataset(variant='100K')

train, test = random_train_test_split(dataset)

model = ExplicitFactorizationModel(n_iter=1)
model.fit(train)

rmse = rmse_score(model, test)
print(rmse)
    
# %%
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate

# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')

# Use the famous SVD algorithm.
algo = SVD()

# Run 5-fold cross-validation and print results.
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
# %%
# using property class
class Celsius:
    def __init__(self, temperature=0):
        self.temperature = temperature

    def to_fahrenheit(self):
        return (self.temperature * 1.8) + 32

    # getter
    def get_temperature(self):
        print("Getting value...")
        return self._temperature

    # setter
    def set_temperature(self, value):
        print("Setting value...")
        if value < -273.15:
            raise ValueError("Temperature below -273.15 is not possible")
        self._temperature = value

    # creating a property object
    temperature = property(get_temperature, set_temperature)

h=Celsius(37)
#%%