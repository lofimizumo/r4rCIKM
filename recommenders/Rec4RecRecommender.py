from recommenders.ISeqRecommender import ISeqRecommender
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from util import evaluation
from random import randint

class Rec4RecRecommender(ISeqRecommender):

    def __init__(self, num_items, dims, rec_ensemble, model_args, pretrained_embeddings=None):

        super().__init__()
        logging.basicConfig(level=logging.DEBUG)

        self.num_items = num_items
        self.config = model_args
        self.logger = logging.getLogger(__name__)
        if pretrained_embeddings is None:
            self.model = MF(20000, model_args)
        else:
            self.model = MF(20000, model_args,
                            pretrained_embeddings=pretrained_embeddings)
        self.ensemble = rec_ensemble
        self.rec_count = len(rec_ensemble)
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

    def fit(self, sequences, metrics):
        sequences = sequences.loc[sequences['sequence'].map(len) > abs(1), 'sequence'].values
        rec_eval_scores = evaluation.predict_score_of_sequences(self.ensemble,
                                                            test_sequences=sequences,
                                                            given_k=1,
                                                            look_ahead=1,
                                                            evaluation_functions=metrics.values(),
                                                            top_n=10,
                                                            scroll=False  
                                                            )
        # build support set for each base recommender here
        
        np_sub_scores = np.array(rec_eval_scores)
        sub_scores = np_sub_scores.sum(0)/np_sub_scores.shape[0]
        print(sub_scores)
        seq, labels_and_negs = self.label(sequences, rec_eval_scores, 5)
        lst = list(seq)
        lst = [list(map(int, i)) for i in lst]
        sequences = np.asarray(lst)
        
        
        sequences_np = sequences
        targets_np = labels_and_negs
        num_items = self.num_items
        n_train = len(sequences_np)
        self.logger.info("Total training records:{}".format(n_train))

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.l2)

        record_indexes = np.arange(n_train)
        batch_size = self.config.batch_size
        num_batches = int(n_train / batch_size) + 1
        for epoch_num in range(self.config.n_iter):
            t1 = time()
            epoch_loss = 0.0
            for batchID in range(num_batches):
                start = batchID * batch_size
                end = start + batch_size

                if batchID == num_batches - 1:
                    if start < n_train:
                        end = n_train
                    else:
                        break

                batch_record_index = record_indexes[start:end]

                batch_sequences = sequences_np[batch_record_index]
                batch_targets = targets_np[batch_record_index]

                prediction_score = self.model(
                    batch_sequences, batch_targets)

                (targets_prediction, negatives_prediction) = torch.split(
                    prediction_score, [1, 1], dim=1)

                loss = -torch.log(torch.sigmoid(targets_prediction -
                                                negatives_prediction) + 1e-8)
                loss = torch.mean(torch.sum(loss))

                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            epoch_loss /= num_batches

            t2 = time()

            output_str = "Epoch %d [%.1f s]  loss=%.4f" % (
                epoch_num + 1, t2 - t1, epoch_loss)
            self.logger.info(output_str)
    
    def label(self, train_data, rec_eval_scores, sequence_length):
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
        ret = np.delete(train_data, rows_to_discard, axis=0)
        labels_and_negs = np.asarray(labels_and_negs)
        return ret, labels_and_negs 


    def recommend(self, item_seq, user_id=None):

        sequence = [int(x) for x in item_seq]
        sequence = torch.LongTensor(sequence).to(self.device)
        emb = self.model.item_emb(sequence)
        emb = emb.sum(0)

        rec_to_predict = np.arange(self.rec_count)
        rec_to_predict = torch.LongTensor(rec_to_predict).to(self.device)
        rec_embs = self.model.rec_emb(rec_to_predict)
        b = self.model.rec_b(rec_to_predict)
        fit_scores = torch.matmul(emb, rec_embs.T)
        most_fit_rec = int(fit_scores.argmax())
        ensemble_output = self.ensemble[most_fit_rec].recommend(item_seq)
        return ensemble_output


class MF(nn.Module):
    def __init__(self, num_items, model_args, pretrained_embeddings=None):
        super(MF, self).__init__()

        self.args = model_args
        self.tf = torch.nn.TransformerEncoderLayer(d_model=100, nhead=4)
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        dims = self.args.d

        self.rec_emb = nn.Embedding(
            100, dims, padding_idx=0).to(self.device)
        self.rec_emb.weight.data.normal_(
            0, 1.0 / self.rec_emb.embedding_dim)
        # Todo: use embedding learnt from support set instead random initialized value here
        
        self.rec_b = nn.Embedding(
            num_items, 1, padding_idx=0).to(self.device)
        self.rec_b.weight.data.zero_()

        if pretrained_embeddings is None:
            self.item_emb = nn.Embedding(
                num_items, dims, padding_idx=0).to(self.device)
            self.rec_b = nn.Embedding(
                num_items, 1, padding_idx=0).to(self.device)

            self.item_emb.weight.data.normal_(
                0, 1.0 / self.item_emb.embedding_dim)
            self.rec_b.weight.data.zero_()
        else:
            self.item_emb = nn.Embedding(
                num_items, dims, padding_idx=0).to(self.device)
            for i in pretrained_embeddings.wv.index2word:
                self.item_emb.weight.data[int(i)] = torch.nn.Parameter(torch.FloatTensor(
                    pretrained_embeddings[i]))

        # self.item_emb.weight.requires_grad = False

    def forward(self, item_seq, rec_to_estimate):
        item_seq = item_seq
        sequence_embs = []
        for i in item_seq:
            i = torch.LongTensor(i).to(self.device)
            embs = self.item_emb(i)
            embs = embs.sum(0)
            sequence_embs.append(embs)

        sequence_embs = torch.stack(sequence_embs)
        sequence_embs = sequence_embs.unsqueeze(dim=0)
        sequence_embs = self.tf(sequence_embs).squeeze(dim=0)

        self.seq_embs = sequence_embs
        rec_to_estimate = torch.LongTensor(rec_to_estimate).to(self.device)
        rec_embs = self.rec_emb(rec_to_estimate)
        b = self.rec_b(rec_to_estimate)
        res = torch.baddbmm(b, rec_embs, sequence_embs.unsqueeze(2)).squeeze()
        return res



