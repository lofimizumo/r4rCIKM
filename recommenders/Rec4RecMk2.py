from recommenders.ISeqRecommender import ISeqRecommender
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from util import evaluation
from random import randint


class R4RRecommender(ISeqRecommender):

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

    def generateBaseEmbeddings(self, sequences, scores, support_size=None):
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
            scores, axis=0), axis=0).transpose()
        ind_best_performed = ind_best_performed[:, :support_size].tolist()
        ret = {}
        for i, indices_rec_k in enumerate(ind_best_performed):
            rec_k_topK = [sequences[index] for index in indices_rec_k]
            ret[i] = rec_k_topK
        return ret

    def fit(self, sequences, metrics):
        """
        fit
        """
        rec_eval_scores = evaluation.predict_score_of_sequences(self.ensemble,
                                                                test_sequences=sequences,
                                                                given_k=1,
                                                                look_ahead=1,
                                                                evaluation_functions=metrics.values(),
                                                                top_n=10,
                                                                scroll=False
                                                                )
        # build support set for each base recommender here
        support = self.generateBaseEmbeddings(sequences, rec_eval_scores, 10)
        self.model.encode_rec(support,range(self.rec_count))

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

    def recommend(self, item_seq, user_id=None, ground_truth=None):

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
        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.pos_emb = torch.nn.Embedding(
            self.args.maxlen, dims)  # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=self.args.dropout_rate)
        self.last_layernorm = torch.nn.LayerNorm(
            self.args.hidden_units, eps=1e-8)

        for _ in range(self.args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(
                self.args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(self.args.hidden_units,
                                                         self.args.num_heads,
                                                         self.args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(
                self.args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(
                self.args.hidden_units, self.args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def encode_seq(self, input_seq):
        """
        multihead-attention to encode the input sequence
        """
        input_seq = np.atleast_2d(input_seq)
        seq = self.item_emb(torch.LongTensor(input_seq))
        seq *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(input_seq.shape[1])), [
                            input_seq.shape[0], 1])
        seq += self.pos_emb(torch.LongTensor(positions))
        seq = self.emb_dropout(seq)

        timeline_mask = torch.BoolTensor(input_seq == 0)
        seq *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seq.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones(
            (tl, tl), dtype=torch.bool))

        for i in range(len(self.attention_layers)):
            seq = torch.transpose(seq, 0, 1)
            Q = self.attention_layernorms[i](seq)
            mha_outputs, _ = self.attention_layers[i](Q, seq, seq,
                                                      attn_mask=attention_mask)
            # key_padding_mask=timeline_mask
            # need_weights=False) this arg do not work?
            seq = Q + mha_outputs
            seq = torch.transpose(seq, 0, 1)

            seq = self.forward_layernorms[i](seq)
            seq = self.forward_layers[i](seq)
            seq *= ~timeline_mask.unsqueeze(-1)

        ret = self.last_layernorm(seq)  # (U, T, C) -> (U, -1, C)

        return ret.squeeze(0).mean(0).squeeze(0)

    def encode_rec(self, support, rec_to_estimate):
        """
        docstring
        """
        seq_embeddings = []

        for rec in rec_to_estimate:
            seqs = support[rec]
            for seq in seqs:
                seq = [int(x) for x in seq]
                seq_embeddings.append(self.encode_seq(seq))
            rec_emb = torch.stack(seq_embeddings).mean(dim=0)
            self.rec_emb.weight.data[rec] = torch.nn.Parameter(torch.FloatTensor(
                rec_emb))

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


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(
            self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        # as Conv1D requires (N, C, Length)
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs
