# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
from kedro.io import DataCatalog
import yaml


class dataset:
    """ 
    dataset: dataset catalog
    """
    
    # initialization function
    def __init__(self, catalog_config_file='../../conf/base/catalog.yml'):
        '''
        Parameters: dataset name: dName, dataset description: dDescription
        Assign the parameters to the entries of the base class
        '''
        

        with open(catalog_config_file, "r") as stream:
            try:
                self.config=yaml.safe_load(stream)
                print(self.config)
            except yaml.YAMLError as exc:
                print(exc)
        self.catalog = DataCatalog.from_config(self.config)
    
    # information print function
    def print_dataset_information(self, dataset_name):
        '''
        Print the basic information about the dataset class
        inclduing the dataset name, and dataset description
        '''
        print('Dataset Name: ' + dataset_name)
        print(self.config[dataset_name])
        
    def load_dataset(self, dataset_name):
        return self.catalog.load(dataset_name)

#     # dataset load abstract function
#     @abc.abstractmethod
#     def load(self):
#         return
    