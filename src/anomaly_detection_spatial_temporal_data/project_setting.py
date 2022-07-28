# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
    Author: Guang Yang
    Reference    - https://github.com/JihoChoi
"""

import sys
import os
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class GDN():
    def __init__(self):
        print("================================")
        print("module checker")
        print("================================")
        print("--------------------------------")

        import torch;
        print("torch version:", torch.__version__)
        print("cuda version:", torch.version.cuda)


        import torch_geometric
        print("torch_geometric version:", torch_geometric.__version__)
        print("--------------------------------")

class TADDY():
    def __init__(self):
        print("================================")
        print("module checker")
        print("================================")
        print("--------------------------------")

        import torch;
        print("torch version:", torch.__version__)
        print("cuda version:", torch.version.cuda)


        #import torch_geometric
        #print("torch_geometric version:", torch_geometric.__version__)
        #print("--------------------------------")
        
        import transformers
        print("Transformer version:", transformers.__version__)
        print("--------------------------------")
        
        import scipy
        print("Scipy version:", scipy.__version__)
        print("--------------------------------")
        import numpy
        print("Numpy version:", numpy.__version__)
        print("--------------------------------")
        import networkx
        print("Networkx version:", networkx.__version__)
        print("--------------------------------")
        import sklearn
        print("Scikit-learn version:", sklearn.__version__)
        print("--------------------------------")
        
#         from torch_scatter import scatter_mean
#         from torch_scatter import scatter_max
#         from torch_geometric.data import DataLoader
#         from torch_geometric.nn import GCNConv

