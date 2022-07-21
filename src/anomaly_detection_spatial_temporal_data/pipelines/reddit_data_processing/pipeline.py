"""
This is a boilerplate pipeline 'reddit_data_processing'
generated using Kedro 0.18.0
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_comment_history

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_comment_history,
                inputs="user_behavior_data",
                #user_label, u2index, p2index, edgelist_df, data_tvt, user2vec_npy, prod2vec_npy
                outputs=[
                    "reddit_processed_node_label", 
                    "reddit_processed_user_id", 
                    "reddit_processed_topic_id",
                    "reddit_processed_edge_list",
                    "reddit_processed_data_split",
                    "reddit_processed_user_feature",
                    "reddit_processed_topic_feature",
                ],
                name="preprocess_reddit_data_node",
            ),
        ]
    )