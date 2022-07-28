# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
    
import sys
import os
import json


def path_join(*elem):
    return os.path.join(*elem)

def ensure_directory(path):
    path = os.path.split(path)
    if not os.path.exists(path[0]):
        os.makedirs(path[0])

def save_json_file(path, data):
    ensure_directory(path)
    with open(path, "w") as json_file:
        json_file.write(json.dumps(data))

def append_json_file(path, data):
    ensure_directory(path)
    with open(path, 'a') as json_file:
        json_file.write(json.dumps(data))

def load_json_file(path):
    with open(path, "r") as json_file:
        data = json.loads(json_file.read())
    return data

def write_data(path, data):
    ensure_directory(path)
    with open(path, "w") as file:
        file.write(data)

def print_dict(dict_file):
    for key in dict_file.keys():
        print("\t {0}: {1}".format(key, dict_file[key]))
    print()
    
def save_to_pickle(file, obj):
    with open(file, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(file):
    with open(file, 'rb') as handle:
        obj = pickle.load(handle)
        return obj

def save_model(model, name, directory):
    with open(os.path.join(directory, name), "wb") as f:
        joblib.dump(model, f)
        
def open_model(name, directory):
    with open(os.path.join(directory, name), "rb") as f:
        model = joblib.load(f)
    return model

class MultipleOptimizer():
    """ a class that wraps multiple optimizers """
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def update_lr(self, op_index, new_lr):
        """ update the learning rate of one optimizer
        Parameters: op_index: the index of the optimizer to update
                    new_lr:   new learning rate for that optimizer """
        for param_group in self.optimizers[op_index].param_groups:
            param_group['lr'] = new_lr
