# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This is a boilerplate pipeline 'taddy'
generated using Kedro 0.18.0
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load_data, set_and_train_model, predict

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_data,
                inputs=["preprocessed_graph_data", "params:data_load_options"],
                outputs="data_dict",
                name="load_processed_data_node",
            ),
            node(
                func=set_and_train_model,
                inputs=["data_dict", "params:model_options"],
                outputs=["train_result",'model_path'],
                name="set_and_train_model_node",
            ),
            node(
                func=predict,
                inputs=["model_path", "params:infer_options"],
                outputs="inference_result",
                name="model_inference_node",
            ),
        ]
    )