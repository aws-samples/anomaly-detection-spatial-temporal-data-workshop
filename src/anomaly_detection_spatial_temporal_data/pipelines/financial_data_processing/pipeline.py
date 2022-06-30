"""
This is a boilerplate pipeline 'financial_data_processing'
generated using Kedro 0.18.0
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_transaction_history, split_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_transaction_history,
                inputs="financial_data",
                outputs=["processed_node_id", "processed_edge_list", "processed_edge_label"],
                name="preprocess_transaction_data_node",
            ),
            node(
                func=split_data,
                inputs=["processed_edge_list","processed_node_id","params:data_process_options"],
                outputs="preprocessed_graph_data",
                name="data_split",
            ),
        ]
    )