"""
This is a boilerplate pipeline 'financial_data_processing'
generated using Kedro 0.18.0
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_transaction_history, construct_graph

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_transaction_history,
                inputs="financial_data",
                outputs="preprocessed_companies",
                name="preprocess_transaction_data_node",
            ),
            node(
                func=construct_graph,
                inputs="shuttles",
                outputs="preprocessed_shuttles",
                name="preprocess_shuttles_node",
            ),
        ]
    )