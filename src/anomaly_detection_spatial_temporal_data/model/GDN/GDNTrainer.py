# -*- coding: utf-8 -*-
from kedro.extras.datasets.text import TextDataSet
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset

from sklearn.preprocessing import MinMaxScaler

from .util.env import get_device, set_device
from .util.preprocess import build_loc_net, construct_data
from .util.net_struct import get_feature_map, get_fc_graph_struc
from .util.iostream import printsep

from .TimeDataset import TimeDataset


from .GDN import GDN

from .train import train
from .test  import test
from .evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores, get_pred_from_scores

import sys
from datetime import datetime

import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import json
import random

class GDNTrainer():
    def __init__(self, sensor_cols, train_orig, test_orig, train_config, env_config, debug=False):
        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

#         dataset = self.env_config['dataset'] 
#         train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
#         test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0)
       
        train, test = train_orig, test_orig

        if 'attack' in train.columns:
            train = train.drop(columns=['attack'])

        feature_map = get_feature_map(sensor_cols)
        fc_struc = get_fc_graph_struc(sensor_cols)

        set_device(env_config['device'])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)

        self.feature_map = feature_map

        train_dataset_indata = construct_data(train, feature_map, labels=0)
        test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())

        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)

        train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'], val_ratio = train_config['val_ratio'])

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset


        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                            shuffle=False, num_workers=0)


        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)

        self.model = GDN(edge_index_sets, len(feature_map), 
                dim=train_config['dim'], 
                input_dim=train_config['slide_win'],
                out_layer_num=train_config['out_layer_num'],
                out_layer_inter_dim=train_config['out_layer_inter_dim'],
                topk=train_config['topk']
            ).to(self.device)



    def run(self):

        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
        else:
            model_save_path = self.get_save_path()[0]

            self.train_log = train(self.model, model_save_path, 
                config = self.train_config,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader, 
                feature_map=self.feature_map,
                test_dataloader=self.test_dataloader,
                test_dataset=self.test_dataset,
                train_dataset=self.train_dataset,
#                 dataset_name=self.env_config['dataset']
            )
        
        # test            
        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)

        _, self.test_result = test(best_model, self.test_dataloader)
        _, self.val_result = test(best_model, self.val_dataloader)

        metrics = self.get_score(self.test_result, self.val_result)
        kedro_text_dataset = TextDataSet(
            filepath=f"{os.path.dirname(model_save_path)}/{os.path.basename(model_save_path)}_metrics.txt"
        )
        kedro_text_dataset.save(metrics)
        

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)


        train_dataloader = DataLoader(train_subset, batch_size=batch,
                                shuffle=True)

        val_dataloader = DataLoader(val_subset, batch_size=batch,
                                shuffle=False)

        return train_dataloader, val_dataloader

    def get_score(self, test_result, val_result):

        feature_num = len(test_result[0][0])
        np_test_result = np.array(test_result)
        np_val_result = np.array(val_result)

        test_labels = np_test_result[2, :, 0].tolist()
    
        test_scores, normal_scores = get_full_err_scores(test_result, val_result)

        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1) 
        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)


        result_str = '=========================** Result **============================\n'
    
        info = None
        if self.env_config['report'] == 'best':
            info = top1_best_info
        elif self.env_config['report'] == 'val':
            info = top1_val_info

        result_str += f'F1 score: {info[0]}\n'
        result_str += f'precision: {info[1]}\n'
        result_str += f'recall: {info[2]}\n'
        return result_str


    def get_save_path(self, feature_name=''):

        dir_path = self.env_config['checkpoint_save_dir']
        
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = self.datestr          

        paths = [
            f'{dir_path}/pretrained/best_{datestr}.pt',
            f'{dir_path}/results/{datestr}.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths
    
    def predict(self):
        best_model = self.model.to(self.device)

        _, self.val_result = test(best_model, self.val_dataloader)
        _, self.test_result = test(best_model, self.test_dataloader)
        
        
        feature_num = len(self.test_result[0][0])
        np_test_result = np.array(self.test_result)
        
        test_labels = np_test_result[2, :, 0].tolist()
        test_scores, normal_scores = get_full_err_scores(self.test_result, self.val_result)
        
        pred, labels = get_pred_from_scores(test_scores, test_labels, topk=1) 
        
        return pred, labels