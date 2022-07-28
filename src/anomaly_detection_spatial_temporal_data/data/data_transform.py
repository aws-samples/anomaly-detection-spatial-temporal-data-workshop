# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import random
import os
import pandas as pd
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix,coo_matrix,eye
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import re
import collections
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import gensim.downloader
#from dataset import dataset 
from anomaly_detection_spatial_temporal_data.utils import ensure_directory
#SAVE_PATH = '../../../../data/02_intermediate/financial_fraud'

class CSVtoTS():
    """Convert raw csv data into format time series models assume"""
    
    def __init__(self, df):
        self.df = df
        
    def split_data_into_separate_ts(self, ts_count_threshold, time_col, value_col, label_col, data_dir, label_dir):
        customer_category_trans_count = self.df.groupby(by=['customer','category']).agg({time_col:'count'})
        customer_category_trans_more_than_threshold =customer_category_trans_count.loc[customer_category_trans_count.step>ts_count_threshold].reset_index()
        customer_category_pairs_for_ts = np.array(customer_category_trans_more_than_threshold[['customer','category']])
        for c_m_p in tqdm(customer_category_pairs_for_ts):
            c_m_p_records = self.df.loc[(self.df.customer==c_m_p[0])&( self.df.category==c_m_p[1])]
            c_m_p_data = c_m_p_records[[time_col,value_col]]
            c_m_p_data.rename(columns={time_col: "timestamp", value_col: "value"}, inplace=True)
            save_file_path = os.path.join(data_dir, '{}_{}_transaction_data.csv'.format(c_m_p[0], c_m_p[1]))
            ensure_directory(save_file_path)
            c_m_p_data.to_csv(save_file_path, index=False)
            #label data 
            c_m_p_label = c_m_p_records[[time_col,label_col]]
            c_m_p_label.rename(columns={time_col: "timestamp", label_col: "label"}, inplace=True)
            label_file_path = os.path.join(label_dir, '{}_{}_transaction_label.csv'.format(c_m_p[0], c_m_p[1]))
            ensure_directory(label_file_path)
            c_m_p_label.to_csv(label_file_path, index=False)
               
class CSVtoDynamicGraphWithNodeFeatures():
    """
    Convert raw csv data into format static graph 
    models assume (e.g. node list and edge list)
    """
    
    def __init__(self, df):
        self.df = df
        nltk.download('punkt')
        nltk.download('stopwords')
        self.w2v_model = gensim.downloader.load('word2vec-google-news-300')
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.stemmer= PorterStemmer()
        
    def _common_member(self, a, b):
        """check common elements of a and b"""
        a_set = set(a)
        b_set = set(b)

        if (a_set & b_set):
            return (a_set & b_set)
        else:
            print("No common elements")
    
    def _filter_data(self):
        #drop records with few votes
        df_score = self.df.drop(self.df[abs(self.df.score) < 10].index)
        counts = df_score['author'].value_counts()
        #Drop user if they have posted less than 10 times
        res = df_score[~df_score['author'].isin(counts[counts < 10].index)]
        #Drop users that are [deleted]
        res = res.drop(res[res.author=='[deleted]'].index)
        benign = pd.DataFrame()
        anomaly = pd.DataFrame()
        benign = benign.append(res)
        benign = benign.drop(benign[benign.score < 10].index)
        anomaly = anomaly.append(res)
        anomaly = anomaly.drop(anomaly[anomaly.score > -10].index)
        #remove overlapping authors
        anomaly_author_names = anomaly.author.unique()
        benign_author_names = benign.author.unique()
        overlap_authors = self._common_member(benign_author_names, anomaly_author_names)
        benign = benign[~benign['author'].isin(overlap_authors)]
        return benign,anomaly
    
    def _get_author_label_and_id(self, benign, anomaly):
        """get author label"""
        anomaly_author_names = anomaly.author.unique()
        benign_author_names = benign.author.unique()
        benign_user_label = pd.DataFrame()
        benign_user_label['author'] = benign_author_names
        benign_user_label['label'] = 0 #0 as benign user
        anomalous_user_label = pd.DataFrame()
        anomalous_user_label['author'] = anomaly_author_names
        anomalous_user_label['label'] = 1
        user_label = pd.concat([benign_user_label, anomalous_user_label])
        total_author_names = benign_author_names.tolist() + anomaly_author_names.tolist()
        total_author_names = sorted(list(set(total_author_names)))
        u2index={}
        count = 0
        for author in total_author_names:
            u2index[author]=count
            count+=1
        return user_label, u2index
    
    def _get_subreddit_id(self, benign, anomaly):
        benign_prod_names = benign.subreddit.unique()
        benign_prod_names = benign_prod_names.tolist()
        anomaly_prod_names = anomaly.subreddit.unique()
        anomaly_prod_names = anomaly_prod_names.tolist()
        total_prod_names = benign_prod_names + anomaly_prod_names
        total_prod_names = sorted(list(set(total_prod_names)))
        p2index={}
        count = 0
        for subreddit in total_prod_names:
            p2index[subreddit]=count
            count+=1
        return p2index
        
    def process_edge_list(self, benign, anomaly):
        """get edge list data from raw df"""
        #benign,anomaly = self._filter_data()
        edgelist_df = benign.append(anomaly, ignore_index=True)
        edgelist_df = edgelist_df.sort_values(by = 'retrieved_on')
        edgelist_df = edgelist_df[['author','subreddit','retrieved_on']]
        return edgelist_df
    
    def _process_node_feature(self, df, id2index, id_col):
        """get word embeddings as node features"""
        final_id2vec_npy = np.zeros((len(id2index), 300))

        for p in id2index:
            related_entries = df.loc[df[id_col] == p]
            row_list = []
            for index, rows in related_entries.iterrows():
                my_list = rows.body
                my_list = my_list.replace('\n'," ")
                my_list = my_list.replace('\t'," ")
                my_list = my_list.lower()
                my_list = ''.join([i for i in my_list if not i.isdigit()])
                my_list = re.sub(r'[^\w\s]', ' ', my_list)
                tokens = word_tokenize(my_list)
                my_list = [i for i in tokens if not i in self.stopwords]
                row_list.append(my_list)

            flat_list = [x for xs in row_list for x in xs]
            counter = collections.Counter(flat_list)
            top10 = counter.most_common(10)
            #print(f'top 10 words for subreddit topic {p} are:', top10)

            final_vectors = np.zeros((10, 300))
            for i, w in enumerate(top10):
                try:
                    embedding = self.w2v_model[w[0]]
                    #embedding = embedding.tolist()
                except:
                    print('no embeddings created for word: {}'.format(w[0]))
                    embedding = np.array([0] * 300)
                final_vectors[i,:]=embedding
            final_embeddings = np.sum(final_vectors, axis=0)
            final_id2vec_npy[id2index[p],:] = final_embeddings
        return final_id2vec_npy
    
    def get_user_feature(self, df,  u2index):
        return self._process_node_feature(df, u2index, 'author')
    
    def get_subreddit_feature(self, df, p2index):
        return self._process_node_feature(df, p2index, 'subreddit')
    
    def process_node_features(self, benign, anomaly, u2index, p2index):
        feature_df = pd.concat([benign, anomaly])
        user2vec_npy = self.get_user_feature(feature_df, u2index)
        prod2vec_npy = self.get_subreddit_feature(feature_df, p2index)
        return user2vec_npy, prod2vec_npy
        
    def _generate_n_lists(self, num_of_lists, num_of_elements, value_from=0, value_to=100):
        s = random.sample(range(value_from, value_to), num_of_lists * num_of_elements)
        return [s[i*num_of_elements:(i+1)*num_of_elements] for i in range(num_of_lists)]    
    
    def train_test_data_split(self, num_of_nodes, train_sample_ratio = 0.8, test_ratio=0.5):
        """get train, validation, test set"""
        num_list = 2
        num_of_elements = num_of_nodes // num_list
        l = self._generate_n_lists(num_list, num_of_elements, 0, num_of_nodes) 
        train_sample_num = int(train_sample_ratio*num_of_elements)
        data_tvt = (np.array(l[0][:train_sample_num]), np.array(l[0][train_sample_num:]), np.array(l[1]))
        
        return data_tvt
    
    def process_data(self):
        benign, anomaly = self._filter_data()
        user_label, u2index = self._get_author_label_and_id(benign,anomaly)
        p2index = self._get_subreddit_id(benign,anomaly)
        edgelist_df = self.process_edge_list(benign,anomaly)
        num_user = len(user_label)
        data_tvt = self.train_test_data_split(num_user)
        user2vec_npy, prod2vec_npy = self.process_node_features(benign, anomaly, u2index, p2index)
        
        return user_label, u2index, p2index, edgelist_df, data_tvt, user2vec_npy, prod2vec_npy
        
class CSVtoDynamicGraph():
    """Convert raw csv data into format dynamic graph
    models assume (e.g. node list and edge list with timestamps or sequence index)
    """
    
    def __init__(self, df):
        self.df = df
        
    def _process_edge_data(self, df: pd.DataFrame):
        """
        Get edge list from transaction history to construct graph
        """
        edges = df[['step','customer','merchant','category','amount','fraud']]
        edges_deduped = edges.drop_duplicates(subset=['customer','merchant'], keep='last', )
        edges_array = np.array(edges_deduped[['customer','merchant']])
        vertexs, edges_1d = np.unique(edges_array, return_inverse=True)
        vertex_to_id = {}
        for i,vertex in enumerate(vertexs):
            vertex_to_id.setdefault(vertex,i)
        vertex_to_id_df = pd.DataFrame.from_dict(
            vertex_to_id,  
            orient='index', 
            columns=['idx']
        ).reset_index().rename(columns={"index": "name"})
        edges_idx = np.reshape(edges_1d, [-1, 2])
  
        return vertex_to_id_df, vertex_to_id, edges_deduped

    def _process_label_data(self, edges_deduped: pd.DataFrame, vertex_to_id: Dict):
        """
        Get transaction label and save
        """
        edge_label_arr = np.zeros([edges_deduped.shape[0], 3], dtype=np.int32)
        #for index, row in tqdm(raw_trans_data.iterrows(), total=raw_trans_data.shape[0]):
        for idx, row in tqdm(edges_deduped.reset_index().iterrows(), total=edges_deduped.shape[0]): #using deduped trans 
            #edge_label.setdefault((vertex_to_id[row['customer']],vertex_to_id[row['merchant']]), []).append(row['fraud'])
            edge_label_arr[idx][0] = vertex_to_id[row['customer']]
            edge_label_arr[idx][1] = vertex_to_id[row['merchant']]
            edge_label_arr[idx][2] = row['fraud']
        return edge_label_arr

    def preprocessDataset(self):
        """pre-process dataset"""
        vertex_to_id_df, vertex_to_id, edges_deduped = self._process_edge_data(self.df)
        edge_label_arr = self._process_label_data(edges_deduped, vertex_to_id)
        edge_label_postprocessed_df = pd.DataFrame(edge_label_arr, columns=['source','target','label'])
        return vertex_to_id_df, edge_label_arr, edge_label_postprocessed_df

    
class DynamicGraph():
    """Class to process node and edge list data"""
    def __init__(self, processed_edge_list: np.ndarray, processed_node_id: pd.DataFrame):
        self.edge_list =  processed_edge_list
        self.node_list = processed_node_id
    
    def generateDataset(self, train_per, snap_size):
        m = len(self.edge_list) #edge number 
        n = len(self.node_list) #node number 
        edge_label_arr = self.edge_list
        train_num = int(np.floor(train_per * m))
        train = edge_label_arr[0:train_num, :] #first half being training samples
        test = edge_label_arr[train_num:, :] #second half being test samples 
        t0 = time.time()
        train_mat = csr_matrix(
            (np.ones([np.size(train, 0)], dtype=np.int32), 
             (train[:, 0], train[:, 1])),
            shape=(n, n)
        )
        train_mat = train_mat + train_mat.transpose()
        train_mat = (train_mat + train_mat.transpose() + eye(n)).tolil()
        headtail = train_mat.rows #store the indexes of edges
        degrees = np.array([len(x) for x in headtail])
        
        train_size = int(len(train) / snap_size + 0.5) #making slices of snapshots
        test_size = int(len(test) / snap_size + 0.5)
        print("Train size:%d  %d  Test size:%d %d" %
              (len(train), train_size, len(test), test_size))
        rows = []
        cols = []
        weis = []
        labs = []
        for ii in range(train_size):
            start_loc = ii * snap_size
            end_loc = (ii + 1) * snap_size

            row = np.array(train[start_loc:end_loc, 0], dtype=np.int32)
            col = np.array(train[start_loc:end_loc, 1], dtype=np.int32)
            lab = np.array(train[start_loc:end_loc, 2], dtype=np.int32)
            wei = np.ones_like(row, dtype=np.int32)

            rows.append(row)
            cols.append(col)
            weis.append(wei) #weights
            labs.append(lab) #label
        print("Training dataset contruction finish! Time: %.2f s" % (time.time()-t0))
        t0 = time.time()
        for i in range(test_size):
            start_loc = i * snap_size
            end_loc = (i + 1) * snap_size

            row = np.array(test[start_loc:end_loc, 0], dtype=np.int32)
            col = np.array(test[start_loc:end_loc, 1], dtype=np.int32)
            lab = np.array(test[start_loc:end_loc, 2], dtype=np.int32)
            wei = np.ones_like(row, dtype=np.int32)

            rows.append(row)
            cols.append(col)
            weis.append(wei)
            labs.append(lab)
        print("Test dataset finish constructing! Time: %.2f s" % (time.time()-t0))  
        return rows,cols,labs,weis,headtail,train_size,test_size,n,m