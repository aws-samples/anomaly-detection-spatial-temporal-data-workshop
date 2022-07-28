# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This is a boilerplate pipeline 'nab_model'
generated using Kedro 0.18.0
"""
import pandas as pd 
import numpy as np 
import os
import logging
from typing import Dict, Tuple

from anomaly_detection_spatial_temporal_data.model.time_series import NABAnomalyDetector


def set_and_run_model(label_dict_file_path, parameters: Dict) -> Tuple:
    """
    Set the model training configurations and train the model 
    Args:
        data_dict: data for the dataloader
        parameters: parameters for model training 
    Returns:
        model training result
    """
    if 'iot' in label_dict_file_path:
        input_dir=parameters['iot_input_dir']
        output_dir=parameters['iot_output_dir']
    elif 'financial' in label_dict_file_path:
        input_dir=parameters['financial_input_dir']
        output_dir=parameters['financial_output_dir']
    else:
        raise NotImplementedError
    
    model_obj = NABAnomalyDetector(
        parameters['model_name'], 
        parameters['model_path'],
        input_dir,
        label_dict_file_path, #hacky way to inject dependency
        output_dir,
        
    )
    model_obj.predict()
    return None
