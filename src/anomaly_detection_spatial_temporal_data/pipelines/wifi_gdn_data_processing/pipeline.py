"""
This is a boilerplate pipeline 'wifi_gdn_data_processing'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_host_data, construct_dataset_aggregate_hosts, subtract_received_from_sent, assign_attack_labels, fillna_and_norm, create_descriptive_column_names

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [   
            # net1 cleaned sent
            node(
                func=clean_host_data,
                inputs=["params:udp_subpopulation_data_path_clean", "params:filename_prefix_sent_sent"],
                outputs="net1_cleaned_sent",
                name="net1_cleaned_sent_node"
            ),
            # net 1 clean rcvd
            node(
                func=clean_host_data,
                inputs=["params:udp_subpopulation_data_path_clean", "params:filename_prefix_sent_rcvd"],
                outputs="net1_cleaned_rcvd",
                name="net1_cleaned_rcvd_node"
            ),
            # net 2 cleaned sent
            node(
                func=clean_host_data,
                inputs=["params:udp_subpopulation_data_path_anom", "params:filename_prefix_sent_sent"],
                outputs="net2_cleaned_sent",
                name="net2_cleaned_sent_node"
            ),
            # net 2 cleaned rcvd
            node(
                func=clean_host_data,
                inputs=["params:udp_subpopulation_data_path_anom", "params:filename_prefix_sent_rcvd"],
                outputs="net2_cleaned_rcvd",
                name="net2_cleaned_rcvd_node"
            ),
            
            # net 1 sent aggregate
            node(
                func=construct_dataset_aggregate_hosts,
                inputs="net1_cleaned_sent",
                outputs="net1_sent_aggregate",
                name="net1_sent_aggregation_node"
            ),
            # net 1 rcvd aggregate
            node(
                func=construct_dataset_aggregate_hosts,
                inputs="net1_cleaned_rcvd",
                outputs="net1_rcvd_aggregate",
                name="net1_rcvd_aggregation_node"
            ),
            # net2 sent aggregate
            node(
                func=construct_dataset_aggregate_hosts,
                inputs="net2_cleaned_sent",
                outputs="net2_sent_aggregate",
                name="net2_sent_aggregation_node"
            ),
            # net2 rcvd aggregate
            node(
                func=construct_dataset_aggregate_hosts,
                inputs="net2_cleaned_rcvd",
                outputs="net2_rcvd_aggregate",
                name="net2_rcvd_aggregation_node"
            ),
            
            # delta packets: sent/received
            # net 1
            node(
                func=subtract_received_from_sent,
                inputs=["net1_sent_aggregate", "net1_rcvd_aggregate"],
                outputs="net1_subtracted",
                name="net1_delta_packets_node"
            ),
            # net 2
            node(
                func=subtract_received_from_sent,
                inputs=["net2_sent_aggregate", "net2_rcvd_aggregate"],
                outputs="net2_subtracted",
                name="net2_delta_packets_node"
            ),
            
            # node to only use top 1200 rows from net1
            
            # assign attack labels
            node(
                func=assign_attack_labels,
                inputs=["net1_subtracted", "net2_subtracted"],
                outputs=["net2_with_anom", "net2_anomaly_times"],
                name="assign_anomaly_node"
            ),
            
            # build train and test: fill na and normalize data with minMax
            node(
                func=fillna_and_norm,
                inputs=["net1_subtracted", "net2_with_anom"],
                outputs=["train", "test"],
                name="normalize_data_node"
            ),
            
            # have better column names 
            node(
                func=create_descriptive_column_names,
                inputs="train",
                outputs=["wifi_data_gdn_train", "wifi_sensor_cols_txt"],
                name="clean_train_column_names_node"
            ),
            node(
                func=create_descriptive_column_names,
                inputs="test",
                outputs=["wifi_data_gdn_test", "throwaway"],
                name="clean_test_column_names_node"
            )
        ]
    )
