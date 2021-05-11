import numpy as np
from recommenders.ISeqRecommender import ISeqRecommender
from util.attention.utils import *
import argparse
import time
from tqdm import tqdm
import torch


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


class AttentionRecommender(ISeqRecommender):
    def __init__(self):
        super().__init__()

    def fit(self, train_data):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', default=None)
        parser.add_argument('--train_dir', default=None)
        parser.add_argument('--batch_size', default=128, type=int)
        parser.add_argument('--lr', default=0.01, type=float)
        parser.add_argument('--maxlen', default=50, type=int)
        parser.add_argument('--hidden_units', default=50, type=int)
        parser.add_argument('--num_blocks', default=2, type=int)
        parser.add_argument('--num_epochs', default=10, type=int)
        parser.add_argument('--num_heads', default=1, type=int)
        parser.add_argument('--dropout_rate', default=0.5, type=float)
        parser.add_argument('--l2_emb', default=0.0, type=float)
        parser.add_argument('--device', default='cpu', type=str)
        parser.add_argument('--inference_only', default=False, type=str2bool)
        parser.add_argument('--state_dict_path', default=None, type=str)
        args = parser.parse_args()
        [user_train, user_valid, user_test, usernum, itemnum] = train_data
        # tail? + ((len(user_train) % args.batch_size) != 0)
        num_batch = len(user_train) // args.batch_size
        cc = 0.0
        for u in user_train:
            cc += len(user_train[u])
        print('average sequence length: %.2f' % (cc / len(user_train)))

        sampler = WarpSampler(user_train, usernum, itemnum,
                              batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
        # no ReLU activation in original SASRec implementation?
        model = SASRec(usernum, itemnum, args).to(args.device)

        for name, param in model.named_parameters():
            try:
                torch.nn.init.xavier_uniform_(param.data)
            except:
                pass  # just ignore those failed init layers

        # this fails embedding init 'Embedding' object has no attribute 'dim'
        # model.apply(torch.nn.init.xavier_uniform_)

        model.train()  # enable model training

        epoch_start_idx = 1
        if args.state_dict_path is not None:
            try:
                model.load_state_dict(torch.load(
                    args.state_dict_path, map_location=torch.device(args.device)))
                tail = args.state_dict_path[args.state_dict_path.find(
                    'epoch=') + 6:]
                epoch_start_idx = int(tail[:tail.find('.')]) + 1
            except:  # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
                print('failed loading state_dicts, pls check file path: ', end="")
                print(args.state_dict_path)
                print(
                    'pdb enabled for your quick check, pls type exit() if you do not need it')
                import pdb
                pdb.set_trace()

        # ce_criterion = torch.nn.CrossEntropyLoss()
        # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
        bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
        adam_optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, betas=(0.9, 0.98))

        T = 0.0
        t0 = time.time()

        with tqdm(total=args.num_epochs) as pbar:

            for epoch in range(epoch_start_idx, args.num_epochs + 1):
                if args.inference_only:
                    break  # just to decrease identition
                # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                for step in range(num_batch):
                    u, seq, pos, neg = sampler.next_batch()  # tuples to ndarray
                    u, seq, pos, neg = np.array(u), np.array(
                        seq), np.array(pos), np.array(neg)
                    pos_logits, neg_logits = model(u, seq, pos, neg)
                    pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                        neg_logits.shape, device=args.device)
                    # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
                    adam_optimizer.zero_grad()
                    indices = np.where(pos != 0)
                    loss = bce_criterion(
                        pos_logits[indices], pos_labels[indices])
                    loss += bce_criterion(neg_logits[indices],
                                          neg_labels[indices])
                    for param in model.item_emb.parameters():
                        loss += args.l2_emb * torch.norm(param)
                    loss.backward()
                    adam_optimizer.step()
                    # expected 0.4~0.6 after init few epochs
                    # print("loss in epoch {} iteration {}: {}".format(
                    # epoch, step, loss.item()))
                pbar.update(1)
        sampler.close()
        print("Done")
        return super().fit(train_data)

    def recommend(self, user_profile, user_id):
        return super().recommend(user_profile, user_id=user_id)


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

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(
            self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(
            args.maxlen, args.hidden_units)  # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(
                args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                         args.num_heads,
                                                         args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(
                args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [
                            log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones(
            (tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                                      attn_mask=attention_mask)
            # key_padding_mask=timeline_mask
            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):  # for training
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits  # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        # only use last QKV classifier, a waste
        final_feat = log_feats[:, -1, :]

        item_embs = self.item_emb(torch.LongTensor(
            item_indices).to(self.dev))  # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # preds # (U, I)
