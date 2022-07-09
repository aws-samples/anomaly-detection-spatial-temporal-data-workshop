"""Reference: https://github.com/yuetan031/TADDY_pytorch"""
import time
import os
import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
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
    
class Taddy(BertPreTrainedModel):
    """TADDY model is based on transformer"""
    learning_record_dict = {}
    load_pretrained_path = ''
    save_pretrained_path = ''
    save_model_path = ''

    def __init__(self, data, config):
        super(Taddy, self).__init__(config)
        #self.args = args
        self.config = config
        self.data = data
        self.transformer = BaseTransformer(config)
        self.cls_y = torch.nn.Linear(config.hidden_size, 1)
        self.init_weights()

    # WL dict
    def WL_setting_init(self, node_list, link_list):
        node_color_dict = {}
        node_neighbor_dict = {}

        for node in node_list:
            node_color_dict[node] = 1
            node_neighbor_dict[node] = {}

        for pair in link_list:
            u1, u2 = pair
            if u1 not in node_neighbor_dict:
                node_neighbor_dict[u1] = {}
            if u2 not in node_neighbor_dict:
                node_neighbor_dict[u2] = {}
            node_neighbor_dict[u1][u2] = 1
            node_neighbor_dict[u2][u1] = 1

        return node_color_dict, node_neighbor_dict

    def compute_zero_WL(self, node_list, link_list):
        WL_dict = {}
        for i in node_list:
            WL_dict[i] = 0
        return WL_dict

    # batching + hop + int + time
    def compute_batch_hop(self, node_list, edges_all, num_snap, Ss, k=5, window_size=1):

        batch_hop_dicts = [None] * (window_size-1)
        s_ranking = [0] + list(range(k+1))

        Gs = []
        for snap in range(num_snap):
            G = nx.Graph()
            G.add_nodes_from(node_list)
            G.add_edges_from(edges_all[snap])
            Gs.append(G)

        for snap in range(window_size - 1, num_snap):
            batch_hop_dict = {}
            # S = Ss[snap]
            edges = edges_all[snap]

            # G = nx.Graph()
            # G.add_nodes_from(node_list)
            # G.add_edges_from(edges)

            for edge in edges:
                edge_idx = str(snap) + '_' + str(edge[0]) + '_' + str(edge[1])
                batch_hop_dict[edge_idx] = []
                for lookback in range(window_size):
                    # s = np.array(Ss[snap-lookback][edge[0]] + Ss[snap-lookback][edge[1]].todense()).squeeze()
                    s = Ss[snap - lookback][edge[0]] + Ss[snap - lookback][edge[1]]
                    s[edge[0]] = -1000 # don't pick myself
                    s[edge[1]] = -1000 # don't pick myself
                    top_k_neighbor_index = s.argsort()[-k:][::-1]

                    indexs = np.hstack((np.array([edge[0], edge[1]]), top_k_neighbor_index))

                    for i, neighbor_index in enumerate(indexs):
                        try:
                            hop1 = nx.shortest_path_length(Gs[snap-lookback], source=edge[0], target=neighbor_index)
                        except:
                            hop1 = 99
                        try:
                            hop2 = nx.shortest_path_length(Gs[snap-lookback], source=edge[1], target=neighbor_index)
                        except:
                            hop2 = 99
                        hop = min(hop1, hop2)
                        batch_hop_dict[edge_idx].append((neighbor_index, s_ranking[i], hop, lookback))
            batch_hop_dicts.append(batch_hop_dict)

        return batch_hop_dicts

    # Dict to embeddings
    def dicts_to_embeddings(self, feats, batch_hop_dicts, wl_dict, num_snap, use_raw_feat=False):

        raw_embeddings = []
        wl_embeddings = []
        hop_embeddings = []
        int_embeddings = []
        time_embeddings = []

        for snap in range(num_snap):

            batch_hop_dict = batch_hop_dicts[snap]

            if batch_hop_dict is None:
                raw_embeddings.append(None)
                wl_embeddings.append(None)
                hop_embeddings.append(None)
                int_embeddings.append(None)
                time_embeddings.append(None)
                continue

            raw_features_list = []
            role_ids_list = []
            position_ids_list = []
            hop_ids_list = []
            time_ids_list = []

            for edge_idx in batch_hop_dict:

                neighbors_list = batch_hop_dict[edge_idx]
                edge = edge_idx.split('_')[1:]
                edge[0], edge[1] = int(edge[0]), int(edge[1])

                raw_features = []
                role_ids = []
                position_ids = []
                hop_ids = []
                time_ids = []

                for neighbor, intimacy_rank, hop, time in neighbors_list:
                    if use_raw_feat:
                        raw_features.append(feats[snap-time][neighbor])
                    else:
                        raw_features.append(None)
                    role_ids.append(wl_dict[neighbor])
                    hop_ids.append(hop)
                    position_ids.append(intimacy_rank)
                    time_ids.append(time)

                raw_features_list.append(raw_features)
                role_ids_list.append(role_ids)
                position_ids_list.append(position_ids)
                hop_ids_list.append(hop_ids)
                time_ids_list.append(time_ids)

            if use_raw_feat:
                raw_embedding = torch.FloatTensor(raw_features_list)
            else:
                raw_embedding = None
            wl_embedding = torch.LongTensor(role_ids_list)
            hop_embedding = torch.LongTensor(hop_ids_list)
            int_embedding = torch.LongTensor(position_ids_list)
            time_embedding = torch.LongTensor(time_ids_list)

            raw_embeddings.append(raw_embedding)
            wl_embeddings.append(wl_embedding)
            hop_embeddings.append(hop_embedding)
            int_embeddings.append(int_embedding)
            time_embeddings.append(time_embedding)

        return raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings

        
    def forward(self, init_pos_ids, hop_dis_ids, time_dis_ids, idx=None):

        outputs = self.transformer(init_pos_ids, hop_dis_ids, time_dis_ids)

        sequence_output = 0
        for i in range(self.config.k+1):
            sequence_output += outputs[0][:,i,:]
        sequence_output /= float(self.config.k+1)

        output = self.cls_y(sequence_output)

        return output

    def batch_cut(self, idx_list):
        batch_list = []
        for i in range(0, len(idx_list), self.config.batch_size):
            batch_list.append(idx_list[i:i + self.config.batch_size])
        return batch_list

    def evaluate(self, trues, preds):
        aucs = {}
        for snap in range(len(self.data['snap_test'])):
            auc = metrics.roc_auc_score(trues[snap],preds[snap])
            aucs[snap] = auc

        trues_full = np.hstack(trues)
        preds_full = np.hstack(preds)
        auc_full = metrics.roc_auc_score(trues_full, preds_full)
        
        return aucs, auc_full

    def generate_embedding(self, edges):
        num_snap = len(edges)
        # WL_dict = compute_WL(self.data['idx'], np.vstack(edges[:7]))
        WL_dict = self.compute_zero_WL(self.data['idx'],  np.vstack(edges[:7]))
        batch_hop_dicts = self.compute_batch_hop(self.data['idx'], edges, num_snap, self.data['S'], self.config.k, self.config.window_size)
        raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = \
            self.dicts_to_embeddings(self.data['X'], batch_hop_dicts, WL_dict, num_snap)
        return raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings

    def negative_sampling(self, edges):
        negative_edges = []
        node_list = self.data['idx']
        num_node = node_list.shape[0]
        for snap_edge in edges:
            num_edge = snap_edge.shape[0]

            negative_edge = snap_edge.copy()
            fake_idx = np.random.choice(num_node, num_edge)
            fake_position = np.random.choice(2, num_edge).tolist()
            fake_idx = node_list[fake_idx]
            negative_edge[np.arange(num_edge), fake_position] = fake_idx

            negative_edges.append(negative_edge)
        return negative_edges

    def train_model(self, max_epoch):

        optimizer = optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=float(self.config.weight_decay))
        raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = self.generate_embedding(self.data['edges'])
        self.data['raw_embeddings'] = None

        ns_function = self.negative_sampling
        auc_full_prev = 0
        for epoch in range(max_epoch):
            t_epoch_begin = time.time()

            # -------------------------
            negatives = ns_function(self.data['edges'][:max(self.data['snap_train']) + 1])
            raw_embeddings_neg, wl_embeddings_neg, hop_embeddings_neg, int_embeddings_neg, \
            time_embeddings_neg = self.generate_embedding(negatives)
            self.train()

            loss_train = 0
            
            for snap in self.data['snap_train']:

                if wl_embeddings[snap] is None:
                    continue
                int_embedding_pos = int_embeddings[snap]
                hop_embedding_pos = hop_embeddings[snap]
                time_embedding_pos = time_embeddings[snap]
                y_pos = self.data['y'][snap].float()

                int_embedding_neg = int_embeddings_neg[snap]
                hop_embedding_neg = hop_embeddings_neg[snap]
                time_embedding_neg = time_embeddings_neg[snap]
                y_neg = torch.ones(int_embedding_neg.size()[0])

                int_embedding = torch.vstack((int_embedding_pos, int_embedding_neg))
                hop_embedding = torch.vstack((hop_embedding_pos, hop_embedding_neg))
                time_embedding = torch.vstack((time_embedding_pos, time_embedding_neg))
                y = torch.hstack((y_pos, y_neg))

                optimizer.zero_grad()

                output = self.forward(int_embedding, hop_embedding, time_embedding).squeeze()
                loss = F.binary_cross_entropy_with_logits(output, y)
                loss.backward()
                optimizer.step()

                loss_train += loss.detach().item()

            loss_train /= len(self.data['snap_train']) - self.config.window_size + 1
            print('Epoch: {}, loss:{:.4f}, Time: {:.4f}s'.format(epoch + 1, loss_train, time.time() - t_epoch_begin))
            self.learning_record_dict.setdefault(epoch + 1, {'train_loss':loss_train})
            if ((epoch + 1) % self.config.print_feq) == 0:
                self.eval()
                preds = []
                for snap in self.data['snap_test']:
                    int_embedding = int_embeddings[snap]
                    hop_embedding = hop_embeddings[snap]
                    time_embedding = time_embeddings[snap]

                    with torch.no_grad():
                        output = self.forward(int_embedding, hop_embedding, time_embedding, None)
                        output = torch.sigmoid(output)
                    pred = output.squeeze().numpy()
                    preds.append(pred)

                y_test = self.data['y'][min(self.data['snap_test']):max(self.data['snap_test'])+1]
                y_test = [y_snap.numpy() for y_snap in y_test]

                aucs, auc_full = self.evaluate(y_test, preds)

                for i in range(len(self.data['snap_test'])):
                    print("Snap: %02d | AUC: %.4f" % (self.data['snap_test'][i], aucs[i]))
                print('TOTAL AUC:{:.4f}'.format(auc_full))
                self.learning_record_dict[epoch + 1].setdefault('test_auc',auc_full)
                
                if auc_full>auc_full_prev:
                    self.save_model_path = os.path.join(self.config.save_directory,f'taddy_model_{epoch}.pth')
                    torch.save(self,self.save_model_path)
    
    def load_model(self, model_path):
        return torch.load(model_path)
    
    def predict(self, snap_num):
        print('Generating embeddings...')
        raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = self.generate_embedding(self.data['edges'])
        print('Embeddings created!')
        self.eval()
        
        #snap = max(self.data['snap_train']) + snap_num
        int_embedding = int_embeddings[snap_num]
        hop_embedding = hop_embeddings[snap_num]
        time_embedding = time_embeddings[snap_num]
        with torch.no_grad():
            output = self.forward(int_embedding, hop_embedding, time_embedding, None)
            output = torch.sigmoid(output)
        pred = output.squeeze().numpy()      
        return pred

    def run(self):
        self.train_model(self.config.max_epoch)
        return self.learning_record_dict, self.save_model_path
    
class Eland():
    def __init__(self):
        pass    