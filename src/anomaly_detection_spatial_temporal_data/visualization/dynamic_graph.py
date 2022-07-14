"""
dynamic_graph.py
----------------
This file holds the main functions for graph visuals
@author Yang, Guang (yaguan@amazon.com)
@date   07/2022
"""

import scipy.sparse as sp
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
import sys, threading
sys.setrecursionlimit(10**7) # max depth of recursion
threading.stack_size(2**27)  # new thread will get stack of such size

from matplotlib import pyplot as plt
import matplotlib
font = {'family' : 'normal', 'size'   : 18}
matplotlib.rc('font', **font)
import matplotlib.cm as cm
plt.rcParams["figure.figsize"] = (30,18)

import pandas as pd
import networkx as nx

NUM_TRANS_PER_SNAPSHOT = 500
NODE_COLOR_DICT = {'customer':'#66a3c7','merchant':'#aea400'}

class DynamicHomoGraph():
    """Class for building and visualizing dynamic homogeneous graph (or bipartite)"""
    
    def __init__(self, node_idx_to_id_mapping, edge_data_w_label, src_node_attribute=None, dst_node_attribute=None, load_time_steps=None):
        """
        Create Dynamic Homogeneous Graph with edge with correlation between objs  
        Arg:
         
          obj_to_type_mapping (pd.DataFrame): object idx to object name mapping obtained from raw csv
          edge_data_w_label(pd.DataFrame): edge data is a dataframe with columns being source node idx, target node idx and edge label
          node_attribute(pd.DataFrame,):  dataframe with node idx and attributes 
          load_time_steps: load how many time window (WARN: load all time steps will take a lot of memory and will take time )

        """
        self.node_idx_to_id_mapping = node_idx_to_id_mapping
        self.edge_data_w_label = edge_data_w_label
        
        if src_node_attribute is not None and dst_node_attribute is not None:
            self.src_node_attribute_names = list(src_node_attribute.columns)
            self.dst_node_attribute_names = list(dst_node_attribute.columns)
            print('Generating node static attribute...')
            self.src_node_static_attribute = self._construct_node_static_features(src_node_attribute, self.src_node_attribute_names, 'customer')
            self.dst_node_static_attribute = self._construct_node_static_features(dst_node_attribute, self.dst_node_attribute_names, 'merchant')
            print('Done!')
        self.timesteps = []
        self.edge_list_snapshot_dict = {}
        self.graph_dict = {}
        
        if load_time_steps is None:
            load_time_steps = edge_data_w_label.shape[0] // NUM_TRANS_PER_SNAPSHOT + 1
        
        for i in range(load_time_steps):
            print(f'Loading time step {i}...\n')
            self.timesteps.append(i)
            edge_list_snapshot = edge_data_w_label[i*NUM_TRANS_PER_SNAPSHOT:(i+1)*NUM_TRANS_PER_SNAPSHOT,:]
            self.edge_list_snapshot_dict.setdefault(i, edge_list_snapshot)
            self.graph_dict.setdefault(i, self.build_graph_from_edge_list(i, edge_list_snapshot))
        
    def _determine_object_type(self, index):
        try: 
            obj_id = self.node_idx_to_id_mapping.loc[self.node_idx_to_id_mapping.idx==index]['name'].values[0]
            if obj_id[1] == 'C':
                obj_type = 'customer'
            elif obj_id[1] == 'M':
                obj_type = 'merchant'
            else:
                obj_type = None
            return obj_type
        except Exception as e:
            print(e)
            print('Did not find such object!')
    
    def _construct_node_list_w_attribute(self, timestep, edge_list_snapshot):
        """build node list with color"""
        node_list = [
            (
                node_idx, 
                {
                    'color':NODE_COLOR_DICT[self._determine_object_type(node_idx)], 
                    'timestep': timestep,
                }
            ) 
            for edge in edge_list_snapshot for node_idx in edge[:2]
        ]

        return node_list
    
    def _construct_node_static_features(self, node_attribute, node_attribute_names, node_id_col):
        """Build a node feature dict from node feature raw dataframe"""
        node_static_attribute = {}
        for i, row in node_attribute.iterrows():
            node_id = row.get(node_id_col)
            for node_attribute_name in node_attribute_names: 
                if node_attribute_name!='step':
                    node_idx = self.node_idx_to_id_mapping.loc[self.node_idx_to_id_mapping.name==node_id]['idx'].values[0]
                    #print(customer_idx, node_attribute_name)
                    node_static_attribute.setdefault(node_idx, {}).setdefault(node_attribute_name, row[node_attribute_name]) 
        return node_static_attribute
        
    def _construct_edge_list_w_label(self, timestep, edge_list_snapshot):
        """build edge list for networkx"""

        edge_list = []
        for edge in edge_list_snapshot:
            edge_list.append((edge[0], edge[1], {'label': edge[2]}))

        return edge_list
        
    def _construct_edge_list_w_weight(self, timestep, edge_list_snapshot):
        """build edge list for networkx"""

        edge_list = []
        for edge in edge_list_snapshot:
            edge_list.append((edge[0], edge[1], {'weight': edge[2]}))

        return edge_list
    
    def build_nx_graph_from_edge_list(self, timestep, edge_list_snapshot):
        nx_g = nx.Graph()
        nx_g.add_nodes_from(
            self._construct_node_list_w_attribute(timestep, edge_list_snapshot), #node list with node 2-tuples of the form (node, node_attribute_dict)
        )
        nx_g.add_edges_from(
            self._construct_edge_list_w_label(timestep, edge_list_snapshot), #edge list with 3-tuple with 2 nodes followed by an edge attribute dictionary, e.g., (2, 3, {'weight': 3.1415})
        )
        return nx_g
    
    def build_graph_from_edge_list(self, timestep, edge_list_snapshot):
        """Creating a NetworkX graph with edge list
        It is recommended to first convert a NetworkX graph into a tuple of node-tensors and then construct a DGLGraph with dgl.graph()."""
        nx_g = self.build_nx_graph_from_edge_list(timestep, edge_list_snapshot)
        return nx_g
               
    def draw_nx_graph_w_edge_label_at_specific_time(self, timestep):
        """only draw nodes with anomaly"""
        fig, ax = plt.subplots(1,1)
        if timestep not in self.timesteps:
            raise AttributeError
        #draw edges with different weights https://networkx.org/documentation/stable/auto_examples/drawing/plot_weighted_graph.html#sphx-glr-auto-examples-drawing-plot-weighted-graph-py
        G = self.graph_dict[timestep]

        # nodes
        node_colors = [d['color'] for (u,d) in G.nodes(data=True)]
        #pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility
        #pos = nx.spiral_layout(G)  # positions for all nodes - seed for reproducibility
        pos = nx.nx_pydot.pydot_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors)
        #edge_weight = [d['weight'] for (u, v, d) in G.edges(data=True)]
        edge_label = [d['label'] for (u, v, d) in G.edges(data=True)]
        # edges
        if len(edge_label)>0:
            neg_edge = [(u, v) for (u, v, d) in G.edges(data=True) if d["label"] == 0]
            pos_edge = [(u, v) for (u, v, d) in G.edges(data=True) if d["label"] == 1]
            nx.draw_networkx_edges(G, pos, edgelist=neg_edge, width=5, alpha=0.8, edge_color="g",)
            nx.draw_networkx_edges(G, pos, edgelist=pos_edge, width=5, alpha=0.8, edge_color="r",)
            

        # node idx
        nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif", font_color='k')

        ax = plt.gca()
        ax.margins(0.08)
        plt.axis("off")
        plt.title('Payment network at timestep {}'.format(timestep))
        plt.tight_layout()
        plt.show()
        
