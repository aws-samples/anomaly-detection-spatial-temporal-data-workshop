# Below code are based on
# https://github.com/yuetan031/TADDY_pytorch
# https://github.com/dm2-nd/eland

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

from anomaly_detection_spatial_temporal_data.model.components import BaseTransformer #components needed for TADDY
from anomaly_detection_spatial_temporal_data.model.components import RGCN_Model #components needed for ELAND

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
    

class Eland_e2e(object):
    """Eland model class based on RGCN"""
    def __init__(self, adj_matrix, lstm_dataloader, user_features, item_features,
            labels, tvt_nids, u2index, p2index, idx2feats, dim_feats=300, cuda=0, hidden_size=128, n_layers=2,
            epochs=400, seed=-1, lr=0.0001, weight_decay=1e-5, dropout=0.4, tensorboard=False,
            log=True, name='debug', gnnlayer_type='gcn', rnn_type='lstm', pretrain_bm=25, pretrain_nc=300, alpha=0.05, bmloss_type='mse', device='cuda', base_pred=400):
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = epochs
        self.pretrain_bm = pretrain_bm
        self.pretrain_nc = pretrain_nc
        self.n_classes = len(np.unique(labels))
        self.alpha = alpha
        self.labels = labels
        self.train_nid, self.val_nid, self.test_nid = tvt_nids
        self.bmloss_type = bmloss_type
        self.base_pred = base_pred
        if log:
            self.logger = self.get_logger(name)
        else:
            self.logger = logging.getLogger()
        # if not torch.cuda.is_available():
        # self.device = torch.device(f'cuda:{cuda}' if cuda >= 0 else 'cpu')
        self.device = device
        # Log parameters for reference
        all_vars = locals()
        self.log_parameters(all_vars)
        # Fix random seed if needed
        if seed > 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # load data
        self.load_data(adj_matrix, user_features, item_features, self.labels, tvt_nids, gnnlayer_type, idx2feats)
        idx2feats = torch.cuda.FloatTensor(idx2feats)
        # idx2feats = idx2feats.to(self.device)
        self.model = RGCN_Model(dim_feats, hidden_size, lstm_dataloader, self.n_classes, n_layers,
                            u2idx = u2index, p2idx = p2index, idx2feats = idx2feats, dropout=dropout,
                            device=self.device, rnn_type=rnn_type , gnnlayer_type=gnnlayer_type,
                            activation=F.relu, bmloss_type=bmloss_type, base_pred=self.base_pred)

    def scipysp_to_pytorchsp(self, sp_mx):
        """ converts scipy sparse matrix to pytorch sparse matrix """
        if not sp.isspmatrix_coo(sp_mx):
            sp_mx = sp_mx.tocoo()
        coords = np.vstack((sp_mx.row, sp_mx.col)).transpose()
        values = sp_mx.data
        shape = sp_mx.shape
        pyt_sp_mx = torch.sparse.FloatTensor(torch.LongTensor(coords.T),
                                             torch.FloatTensor(values),
                                             torch.Size(shape))
        return pyt_sp_mx

    def load_data(self, adj_matrix, user_features, item_features, labels, tvt_nids, gnnlayer_type, idx2feats):
        """Process data"""
        if isinstance(user_features, torch.FloatTensor):
            self.user_features = user_features
        else:
            self.user_features = torch.FloatTensor(user_features)

        if isinstance(item_features, torch.FloatTensor):
            self.item_features = item_features
        else:
            self.item_features = torch.FloatTensor(item_features)

        # Normalize
        self.user_features = F.normalize(self.user_features, p=1, dim=1)
        self.item_features = F.normalize(self.item_features, p=1, dim=1)

        if isinstance(labels, torch.LongTensor):
            self.labels = labels
        else:
            self.labels = torch.LongTensor(labels)

        assert sp.issparse(adj_matrix)
        if not isinstance(adj_matrix, sp.coo_matrix):
            adj_matrix = sp.coo_matrix(adj_matrix)
        self.adj = self.scipysp_to_pytorchsp(adj_matrix).to_dense()

    def pretrain_bm_net(self, n_epochs=25):
        """ pretrain the behavioral modelling network """
        optimizer = torch.optim.Adam(self.model.bm_net.parameters(), lr = self.lr*5)
        if self.bmloss_type == 'mse':
            criterion = MSELoss()
        elif self.bmloss_type == 'cos':
            criterion = CosineEmbeddingLoss()
        self.model.bm_net.train()
        self.model.bm_net.to(self.device)
        for epoch in range(n_epochs):
            self.model.bm_net.zero_grad()
            optimizer.zero_grad()
            cur_loss = []
            for batch_idx, (uids, feats, _, feats_len) in enumerate(self.model.loader):
                feats = feats.to(self.device).float()
                loss = 0
                out, out_len = self.model.bm_net(feats, feats_len)
                for idx in np.arange(len(out_len)):
                    if self.bmloss_type == 'cos':
                        # loss += criterion(out[idx, :out_len[idx]-1, :], feats[idx, 1:out_len[idx], :], torch.cuda.LongTensor([1]))
                        loss += criterion(out[idx, :out_len[idx]-1, :], feats[idx, 1:out_len[idx], :], torch.LongTensor([1]).to(self.device))
                    else:
                        loss += criterion(out[idx, :out_len[idx]-1, :], feats[idx, 1:out_len[idx], :])
                # print('--------')
                # print(torch.isnan(out[idx, :out_len[idx]-1, :]).sum(), torch.isnan(feats[idx, :out_len[idx]-1, :]).sum())
                # print(torch.isnan(out).sum(), torch.isnan(feats).sum())
                # print(loss)
                loss.backward()
                cur_loss.append(loss.item())
                nn.utils.clip_grad_norm_(self.model.bm_net.parameters(), 5)
                optimizer.step()
                optimizer.zero_grad()
                self.model.bm_net.zero_grad()
            self.logger.info(f'BM Module pretrain, Epoch {epoch+1}/{n_epochs}: loss {round(np.mean(cur_loss), 8)}')

    def pretrain_nc_net(self, n_epochs=300):
        """ pretrain the node classification network """
        optimizer = torch.optim.Adam(self.model.nc_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_val_auc = 0.
        best_test_auc = 0.
        best_res = None
        criterion = F.nll_loss
        self.model.nc_net.to(self.device)
        self.user_features = self.user_features.to(self.device)
        self.item_features = self.item_features.to(self.device)
        self.labels = self.labels.to(self.device)

        cnt_wait = 0
        patience = 50
        for epoch in range(n_epochs):
            self.model.nc_net.train()
            self.model.nc_net.zero_grad()
            input_adj = self.adj.clone()
            input_adj = input_adj.to(self.device)
            nc_logits, _ = self.model.nc_net(input_adj, self.user_features, self.item_features)
            loss = criterion(nc_logits[self.train_nid], self.labels[self.train_nid])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Detach from computation graph
            self.adj = self.adj.detach()
            # Validation
            self.model.nc_net.eval()
            with torch.no_grad():
                input_adj = self.adj.clone()
                input_adj = input_adj.to(self.device)
                nc_logits_eval, _ = self.model.nc_net(input_adj, self.user_features, self.item_features)
            res_training = self.eval_node_cls(nc_logits[self.train_nid].detach(), self.labels[self.train_nid], self.n_classes)
            res = self.eval_node_cls(nc_logits_eval[self.val_nid], self.labels[self.val_nid], self.n_classes)

            if res['auc'] > best_val_auc:
                cnt_wait = 0
                best_val_auc = res['auc']
                test_auc = self.eval_node_cls(nc_logits_eval[self.test_nid], self.labels[self.test_nid], self.n_classes)['auc']
                if test_auc > best_test_auc:
                    best_test_auc = test_auc
                    best_res = self.eval_node_cls(nc_logits_eval[self.test_nid], self.labels[self.test_nid], self.n_classes)
                self.logger.info('NCNet pretrain, Epoch [{} / {}]: loss {:.4f}, training auc: {:.4f}, val_auc {:.4f}, test auc {:.4f}'
                        .format(epoch+1, n_epochs, loss.item(), res_training['auc'], res['auc'], test_auc))
            else:
                cnt_wait += 1
                self.logger.info('NCNet pretrain, Epoch [{} / {}]: loss {:.4f}, training auc: {:.4f}, val_auc {:.4f}'
                        .format(epoch+1, n_epochs, loss.item(), res_training['auc'], res['auc']))

            if cnt_wait >= patience:
                self.logger.info('Early stop!')
                break
        self.logger.info('Best Test Results: auc {:.4f}, ap {:.4f}, f1 {:.4f}'.format(best_res['auc'], best_res['ap'], best_res['f1']))

        return best_res['auc'], best_res['ap']

    def train(self):
        """ End-to-end training for bm_net and nc_net """
        # For debugging
        torch.autograd.set_detect_anomaly(True)
        # Move variables to device if haven't done so
        self.user_features = self.move_to_cuda(self.user_features, self.device)
        self.item_features = self.move_to_cuda(self.item_features, self.device)
        self.labels = self.move_to_cuda(self.labels, self.device)
        self.model = self.model.to(self.device)
        # Pretrain
        if self.pretrain_bm > 0:
            self.pretrain_bm_net(self.pretrain_bm)
        if self.pretrain_nc > 0:
            self.pretrain_nc_net(self.pretrain_nc)
        # optimizers
        optims = MultipleOptimizer(torch.optim.Adam(self.model.bm_net.parameters(), lr=self.lr),
                                torch.optim.Adam(self.model.nc_net.parameters(), lr=self.lr, weight_decay=self.weight_decay))

        criterion = F.nll_loss
        best_test_auc = 0.
        best_val_auc = 0.
        best_res = None
        cnt_wait = 0
        patience = 70
        # Training...
        for epoch in range(self.n_epochs):
            self.model.train()
            self.model.zero_grad()
            input_adj = self.adj.clone()
            input_adj = input_adj.to(self.device)
            nc_logits, modified_adj, bm_loss = self.model(input_adj, self.user_features, self.item_features, self.n_epochs, epoch)
            loss = nc_loss = criterion(nc_logits[self.train_nid], self.labels[self.train_nid])
            loss += bm_loss * self.alpha
            optims.zero_grad()
            loss.backward()
            optims.step()
            # Computation Graph
            # Validation
            self.model.eval()
            with torch.no_grad():
                input_adj = self.adj.clone()
                input_adj = input_adj.to(self.device)
                # nc_logits_eval_original, _ = self.model.nc_net(input_adj, self.user_features, self.item_features)
                input_adj = self.adj.clone()
                input_adj = input_adj.to(self.device)
                nc_logits_eval_modified, _, _ = self.model(input_adj, self.user_features, self.item_features, self.n_epochs, epoch)
            training_res = self.eval_node_cls(nc_logits[self.train_nid].detach(), self.labels[self.train_nid], self.n_classes)
            # res = self.eval_node_cls(nc_logits_eval_original[self.val_nid], self.labels[self.val_nid], self.n_classes)
            res_modified = self.eval_node_cls(nc_logits_eval_modified[self.val_nid], self.labels[self.val_nid], self.n_classes)
            if res_modified['auc'] > best_val_auc:
                cnt_wait = 0
                best_val_auc = res_modified['auc']
                # res_test = self.eval_node_cls(nc_logits_eval_original[self.test_nid], self.labels[self.test_nid], self.n_classes)
                res_test_modified = self.eval_node_cls(nc_logits_eval_modified[self.test_nid], self.labels[self.test_nid], self.n_classes)
                if res_test_modified['auc'] > best_test_auc:
                    best_test_auc = res_test_modified['auc']
                    best_res = res_test_modified
                self.logger.info('Eland Training, Epoch [{}/{}]: loss {:.4f}, train_auc: {:.4f}, val_auc {:.4f}, test_auc {:.4f}'
                        .format(epoch+1, self.n_epochs, loss.item(), training_res['auc'], res_modified['auc'], res_test_modified['auc']))
            else:
                cnt_wait += 1
                self.logger.info('Eland Training, Epoch [{}/{}]: loss {:.4f}, train_auc: {:.4f}, val_auc {:.4f}'
                        .format(epoch+1, self.n_epochs, loss.item(), training_res['auc'], res_modified['auc']))

            if cnt_wait >= patience:
                self.logger.info('Early stop!')
                break
        self.logger.info('Best Test Results: auc {:.4f}, ap {:.4f}, f1 {:.4f}'.format(best_res['auc'], best_res['ap'], best_res['f1']))

        return best_res['auc'], best_res['ap']

    def log_parameters(self, all_vars):
        del all_vars['self']
        del all_vars['adj_matrix']
        del all_vars['user_features']
        del all_vars['item_features']
        del all_vars['labels']
        del all_vars['tvt_nids']
        del all_vars['lstm_dataloader']
        del all_vars['u2index']
        del all_vars['p2index']
        del all_vars['idx2feats']
        self.logger.info(f'Parameters: {all_vars}')

    @staticmethod
    def transform_mat(matrix):
        """
            Since in the original matrix, there are items that have zero degree, we add a small delta in order to calculate the norm properly
        """
        delta = 1e-5
        matrix = matrix + delta
        return matrix

    @staticmethod
    def move_to_cuda(var, device):
        if not var.is_cuda:
            return var.to(device)
        else:
            return var

    @staticmethod
    def get_logger(name):
        logger = logging.getLogger(name)
        if (logger.hasHandlers()):
            logger.handlers.clear()
        logger.setLevel(logging.INFO)
        # Foramtter
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        # console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # create file handler
        if name is not None:
            fh = logging.FileHandler(f'logs/ELANDe2e-{name}.log')
            fh.setFormatter(formatter)
            fh.setLevel(logging.INFO)
            logger.addHandler(fh)
        return logger

    @staticmethod
    def eval_node_cls(logits, labels, n_classes):
        logits = logits.cpu().numpy()
        y_pred = np.argmax(logits, axis=1)
        logits = logits.T[1]
        labels = labels.cpu().numpy()

        # fpr, tpr, _ = roc_curve(labels, logits, pos_label=1)
        roc_auc = roc_auc_score(labels, logits)
        # precisions, recalls, _ = precision_recall_curve(labels, logits, pos_label=1)
        ap = average_precision_score(labels, logits, pos_label = 1)
        f1 = f1_score(labels, y_pred)
        conf_mat = np.zeros((n_classes, n_classes))
        results = {
            'f1': f1,
            'ap': ap,
            'conf': conf_mat,
            'auc': roc_auc
        }

        return results