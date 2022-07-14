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


def set_and_run_model(parameters: Dict) -> Tuple:
    """
    Set the model training configurations and train the model 
    Args:
        data_dict: data for the dataloader
        parameters: parameters for model training 
    Returns:
        model training result
    """
    model_obj = NABAnomalyDetector(
        parameters['model_name'], 
        parameters['model_path'],
        parameters['input_dir'],
        parameters['output_dir'],
        
    )
    model_obj.predict()
    return None
