# Some codes below are based on
# https://github.com/yuetan031/TADDY_pytorch
# https://github.com/dm2-nd/eland

import os
import sys
import json
import math
import logging
import pickle as pk
from collections import Counter
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix
import torch
#from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import MSELoss, CosineEmbeddingLoss

from transformers.modeling_bert import BertAttention, BertIntermediate, BertOutput, BertPreTrainedModel, BertPooler
from transformers.configuration_utils import PretrainedConfig


class TransformerEncoder(nn.Module):
    """Encoder for Taddy Transformer"""
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs

class EdgeEncoding(nn.Module):
    """edge encoding for Taddy"""
    def __init__(self, config):
        super(EdgeEncoding, self).__init__()
        self.config = config

        self.inti_pos_embeddings = nn.Embedding(config.max_inti_pos_index, config.hidden_size)
        self.hop_dis_embeddings = nn.Embedding(config.max_hop_dis_index, config.hidden_size)
        self.time_dis_embeddings = nn.Embedding(config.max_hop_dis_index, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=float(config.layer_norm_eps))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, init_pos_ids=None, hop_dis_ids=None, time_dis_ids=None):

        position_embeddings = self.inti_pos_embeddings(init_pos_ids)
        hop_embeddings = self.hop_dis_embeddings(hop_dis_ids)
        time_embeddings = self.hop_dis_embeddings(time_dis_ids)

        embeddings = position_embeddings + hop_embeddings + time_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs

class BaseTransformer(BertPreTrainedModel):
    data = None

    def __init__(self, config):
        super(BaseTransformer, self).__init__(config)
        self.config = config

        self.embeddings = EdgeEncoding(config)
        self.encoder = TransformerEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.raw_feature_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.raw_feature_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def setting_preparation(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(torch.long)  # not converting to long will cause errors with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        return token_type_ids, extended_attention_mask, encoder_extended_attention_mask, head_mask


    def forward(self, init_pos_ids, hop_dis_ids, time_dis_ids, head_mask=None):
        if head_mask is None:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(init_pos_ids=init_pos_ids,
                                           hop_dis_ids=hop_dis_ids, time_dis_ids=time_dis_ids)
        encoder_outputs = self.encoder(embedding_output, head_mask=head_mask) #这里的输出是tuple，因为在某些设定下要输出别的信息（中间分析用）
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs

    def run(self):
        pass

class RGCN_Model(nn.Module):
    def __init__(self, dim_feats, dim_h, lstm_dataloader, n_classes, n_layers, activation,
                dropout, device, gnnlayer_type, u2idx, p2idx, idx2feats, bmloss_type, rnn_type, base_pred):
        super(RGCN_Model, self).__init__()

        self.device = device
        self.gnnlayer_type = gnnlayer_type
        self.loader = lstm_dataloader
        self.u2idx, self.p2idx = u2idx, p2idx
        self.idx2feats = idx2feats.to(self.device)
        self.base_pred = base_pred

        # Behavior Modelling
        self.bm_net = GAU_E(dim_feats, dim_h, idx2feats, p2idx, rnn_type=rnn_type, out_sz = 300)

        # Node Classification
        self.nc_net = GNN(dim_feats, dim_h, n_classes, n_layers, activation, dropout, gnnlayer_type=gnnlayer_type)

        # bm_loss
        self.bmloss_type = bmloss_type
        if bmloss_type == 'mse':
            self.criterion = MSELoss()
        elif bmloss_type == 'cos':
            self.criterion = CosineEmbeddingLoss()

    def forward(self, original_adj, user_features, item_features, total_epochs=100, cur_epoch=100):
        # Behavior Modelling as Graph Augmentation through Delta
        # TODO: Potential improvement
        bm_loss = 0 # init loss for future backward
        for batch_idx, (uids, feats, _, feats_length) in enumerate(self.loader):
            feats = feats.to(self.device).float()
            num_pred = torch.mul(torch.true_divide(feats_length, self.loader.dataset.total_edges), self.base_pred * 10)
            num_pred = torch.floor(num_pred)
            num_pred = num_pred.to(self.device)

            # Using next element as label
            out, out_len = self.bm_net(feats, feats_length)
            for idx in np.arange(len(out_len)):
                if self.bmloss_type == 'cos':
                    bm_loss += self.criterion(out[idx, :out_len[idx]-1, :], feats[idx, 1:out_len[idx], :], torch.cuda.LongTensor([1]))
                else:
                    bm_loss += self.criterion(out[idx, :out_len[idx]-1, :], feats[idx, 1:out_len[idx], :])

            #delta: (batch_size, 1, feature_size)
            delta = out[np.arange(len(out_len)), out_len-1, None]
            delta = delta.squeeze()

            # Store intermediary vars
            feats2 = feats.clone()

            for i in range(1, torch.max(num_pred).int()+1):
                u_delta, pred_features = self.match(delta, 0.5 + 4.5 * (total_epochs-cur_epoch)/total_epochs)
                tmp = i <= num_pred
                # indices = [self.u2idx[uid.item()] for uid in uids]
                indices = [self.u2idx[uid] for uid in uids]
                # Update Graph
                original_adj[indices] += torch.mul(tmp.unsqueeze(1).repeat(1, u_delta.size(1)), u_delta)
                if max(feats_length) >= feats.size(1):
                    if 'cuda' in self.device:
                        feats2 = torch.cat((feats, torch.cuda.FloatTensor(feats.size(0), 1, feats.size(2)).fill_(0.).to(self.device)), dim=1)
                    else:
                        feats2 = torch.cat((feats, torch.FloatTensor(feats.size(0), 1, feats.size(2)).fill_(0.).to(self.device)), dim=1)
                for idx in range(len(feats_length)):
                    feats2[idx][feats_length[idx]] = pred_features.detach()[idx]
                out, out_len = self.bm_net(feats2, feats_length)
                delta = out[np.arange(len(out_len)), out_len-1, None]
                #delta: (batch_size, 1, feature_size)
                delta = delta.squeeze()
        # Update features
        user_features = original_adj @ item_features
        nc_logits, _ = self.nc_net(original_adj, user_features, item_features)

        return nc_logits, original_adj, bm_loss

    def match(self, x, tau):
        """
            x: (batch_size, features_size)
        """
        # match: delta: (batch_size, feature_size) --> (batch_size, dict_size)
        # pred_features: (batch_size, feature_size) --> (batch_size, feature_size)
        similarity_matrix = self.cosine_similarity(x, self.idx2feats)  # idx2feats: (dict_sz, feat_sz)
        similarity_matrix = F.gumbel_softmax(similarity_matrix, tau=tau, hard=True, dim=1)
        pred_features = self.idx2feats[torch.argmax(similarity_matrix, dim=1)]

        return similarity_matrix, pred_features

    @staticmethod
    def min_max(mat):
        return (mat - torch.min(mat, dim=1)[0].reshape(-1, 1)) / (torch.max(mat, dim=1)[0] - torch.min(mat, dim=1)[0]).reshape(-1, 1)

    @staticmethod
    def cosine_similarity(x1, x2):
        """
            x1: (batch_size, feature_size); x2: (dict_size, feature_size)
        """
        x2 = x2.T
        return (x1@x2) / ((torch.norm(x1, p=2, dim=1).reshape(-1, 1) @ torch.norm(x2, p=2, dim=0).reshape(1, -1)) + 1e-8)
    
class GAU_E(nn.Module):
    def __init__(self, dim_feats, dim_h, idx2feats, p2idx, out_sz = 300, rnn_type='lstm', dropout=0.2):
        super(GAU_E, self).__init__()

        self.dim_feats = dim_feats
        self.dim_h = dim_h
        self.out_sz = out_sz
        self.idx2feats = idx2feats
        self.p2idx = p2idx
        # Transform hidden space to feature space
        self.fc = nn.Linear(dim_h, out_sz)
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=dim_feats, hidden_size=dim_h, num_layers=2, bias=True, batch_first=True, dropout=dropout)
        if rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=dim_feats, hidden_size=dim_h, num_layers=2, bias=True, batch_first=True, dropout=dropout)
        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=dim_feats, hidden_size=dim_h, num_layers=2, bias=True, batch_first=True, dropout=dropout)
        self.rnn_type = rnn_type

    def forward(self, feats, feats_length):
        """
            pids: Not used in this verison
            feats: (batch_size, max_len_in_batch, (feat_sz))
            return: (batch_size, max_len_in_batch, (feat_sz))
        """
        sort = np.argsort(-feats_length)
        length_sort = feats_length[sort]
        reversed_sort = np.argsort(sort)

        # packing operation
        x = torch.nn.utils.rnn.pack_padded_sequence(feats[sort], length_sort, batch_first=True)
        if self.rnn_type == 'lstm':
            output, (h, c) = self.rnn(x)
        else:
            output, h = self.rnn(x)
        # unpacking operation
        output, out_len = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # Now we have output of shape (batch_size, len_seq, hidden_size); and we get the last hidden layer
        # output = output[np.arange(len(out_len)), out_len-1, None]
        output = output[reversed_sort]
        out_len = out_len[reversed_sort]

        # Map it to feature space
        output = self.fc(output)

        return output, out_len

    @staticmethod
    def min_max(arr):
        """
            arr: Tensor
        """
        return (arr - torch.min(arr)) / (torch.max(arr)-torch.min(arr))

class GNN(nn.Module):
    """
        GCN and GSage
    """
    def __init__(self, dim_feats, dim_h, n_classes, n_layers,
                activation, dropout, gnnlayer_type='gcn'):
        super(GNN, self).__init__()
        heads = [1] * (n_layers + 1)
        if gnnlayer_type == 'gcn':
            gnnlayer = GCNLayer
        elif gnnlayer_type == 'gsage':
            gnnlayer = SAGELayer
        elif gnnlayer_type == 'hetgcn':
            gnnlayer = HetLayer
        self.gnnlayer_type = gnnlayer_type
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(gnnlayer(dim_feats, dim_h, heads[0], activation, 0))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(gnnlayer(dim_h * heads[i], dim_h, heads[i+1], activation, dropout))
        # output layer
        self.layers.append(gnnlayer(dim_h * heads[-2], n_classes, heads[-1], None, dropout))

    def forward(self, adj, features_u, features_v):
        h_u, h_v = features_u, features_v
        if self.gnnlayer_type == 'gcn':
            d_u, d_v = self.get_normed_d(adj)
            for i, layer in enumerate(self.layers):
                h_u, h_v = layer(adj, h_u, h_v, d_u, d_v)
                if i == len(self.layers) - 2:
                    emb = h_u
        if self.gnnlayer_type == 'gsage' or self.gnnlayer_type == 'hetgcn':
            for i, layer in enumerate(self.layers):
                h_u, h_v = layer(adj, h_u, h_v)
                if i == len(self.layers) - 2:
                    emb = h_u
        # We only need user predictions in the end
        # return h_u
        return F.log_softmax(h_u, dim=1), emb

    @staticmethod
    def get_normed_d(A):
        """ Get normalized degree matrix of A"""
        d_u = A.sum(1) + 1
        # Self Loop
        d_v = A.sum(0) + 1

        d_u = torch.pow(d_u, -0.5)
        d_v = torch.pow(d_v, -0.5)

        return d_u, d_v    

class GCNLayer(nn.Module):
    """ one layer of GCN """
    def __init__(self, input_dim, output_dim, n_heads, activation, dropout, bias=True):
        super(GCNLayer, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.activation = activation
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.b = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    # In order to save GPU memory, we pass n by m matrix instead of bipartite matrix here
    def forward(self, adj, h_u, h_v, D_u, D_v):
        """
            adj: (n, m) tensor
            h_u: (n, f) tensor; user features
            h_v: (m, f) tensor; item features
            D_u: Normed Degree matrix of U
            D_v: Normed Degree matrix of V
        """
        if self.dropout:
            h_u = self.dropout(h_u)
            h_v = self.dropout(h_v)
        x_u = h_u @ self.W
        x_v = h_v @ self.W

        x_u = x_tmp_u = x_u * D_u.unsqueeze(1)
        x_v = x_tmp_v = x_v * D_v.unsqueeze(1)

        x_u = adj @ x_v + x_tmp_u
        x_v = adj.T @ x_tmp_u + x_tmp_v

        x_u = x_u * D_u.unsqueeze(1)
        x_v = x_v * D_v.unsqueeze(1)

        if self.b is not None:
            x_u += self.b
            x_v += self.b

        if self.activation is not None:
            x_u = self.activation(x_u)
            x_v = self.activation(x_v)

        return x_u, x_v