# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This is a boilerplate pipeline 'gdn'
generated using Kedro 0.18.1
"""

import pandas as pd
from typing import Dict, Tuple, List

from anomaly_detection_spatial_temporal_data.model.GDN.GDNTrainer import GDNTrainer


def set_and_train_model(sensor_cols: List[str], train_df: pd.DataFrame, test_df: pd.DataFrame, train_config: Dict, env_config: Dict) -> Tuple:
    # kedro will load text file and leave it as string
    if isinstance(sensor_cols, str):
        sensor_cols = sensor_cols.split("\n")
    trainer = GDNTrainer(sensor_cols, train_df, test_df, train_config, env_config)
    trainer.run()