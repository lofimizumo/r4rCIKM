# %%
import numpy as np

# %%


def generateBaseEmbeddings(sequences, scores):
    """
        input:
        ----------------
        scores:[[1,0,0,1],[0,0,0,1]]
        rec0:[1,0,0,1]
        rec1:[0,0,0,1]
        ----------------
        return: {'rec_k': top_performed_sequences}
        """
    ind_best_performed = np.flip(np.argsort(
        scores, axis=1), axis=1).tolist()
    ret = {}
    for i, indices_rec_k in enumerate(ind_best_performed):
        rec_k_topK = [sequences[index] for index in indices_rec_k]
        ret[i] = rec_k_topK
    return ret


sequences = [[1, 4, 5, 6], [1, 2, 3, 4], [3, 54, 3, 24], [
    34, 2, 1, 4, 5, 6, 7], [23, 5, 6, 7, 5, 3], [3, 4, 2, 3, 1, 1, 2]]
scores = [[0, 1, 1, 0, 0, 1], [0, 1, 0, 0, 0, 1],
          [1, 0, 1, 1, 0, 0], [1, 1, 0, 0, 1, 1]]
np_scores = np.asarray(scores)
generateBaseEmbeddings(sequences, np_scores)

# %%


def embedItemSeq(sequence):
    """
    sequence:Tensor
    return: Tensor
    """
    