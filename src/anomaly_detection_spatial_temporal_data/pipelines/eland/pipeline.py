# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This is a boilerplate pipeline 'eland'
generated using Kedro 0.18.0
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load_data, set_and_train_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_data,
                inputs=[
                    "reddit_processed_node_label", 
                    "reddit_processed_user_id", 
                    "reddit_processed_topic_id",
                    "reddit_processed_edge_list",
                    "reddit_processed_data_split",
                    "reddit_processed_user_feature",
                    "reddit_processed_topic_feature",
                    "params:eland_data_load_options"
                ],
                outputs="reddit_data_dict",
                name="load_processed_data_node",
            ),
            node(
                func=set_and_train_model,
                inputs=["reddit_data_dict", "params:eland_model_options"],
                outputs=["reddit_train_result",'reddit_model_path'],
                name="set_and_train_model_node",
            ),
        ]
    )