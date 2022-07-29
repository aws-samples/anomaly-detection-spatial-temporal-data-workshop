# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import random
import os
import re
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix,coo_matrix,eye
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import collections
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import gensim.downloader

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