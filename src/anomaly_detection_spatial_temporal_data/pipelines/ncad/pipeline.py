# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This is a boilerplate pipeline 'ncad'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load_dataset, standardize_dataset, batadal_inject_anomalies, split_test_into_val, construct_data_module, set_and_train_model, set_up_callbacks, load_and_evaluate

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # load node
            node(
                func=load_dataset,
                inputs=["iot_data_ncad_train", "iot_data_ncad_train_anom", "iot_data_ncad_test", "iot_sensor_cols_txt", "params:label_column_name_ncad"],
                outputs=["train", "test"],
                name="load_dataset_node"
            ),
            # standardize node
            node(
                func=standardize_dataset,
                inputs=["train", "test"],
                outputs=["train_std", "test_std", "ts_channels"],
                name="standardize_dataset_node"
            ),
            # inject anomalies node
            node(
                func=batadal_inject_anomalies,
                inputs=["train_std", "params:injection_method", "params:ratio_injected_spikes"],
                outputs="train_transformed",
                name="inject_anomalies_node"
            ),
            # trian/test split node
            node(
                func=split_test_into_val,
                inputs=["test_std", "params:window_length", "params:suspect_window_length"],
                outputs=["val_transformed", "test_transformed"],
                name="transform_test_node"
            ),
            # NCCAD Data module node
            node(
                func=construct_data_module,
                inputs=[
                    "train_transformed", "val_transformed", "test_transformed", 
                    "params:window_length", "params:suspect_window_length", "params:num_series_in_train_batch",
                    "params:num_crops_per_series", "params:stride_roll_pred_val_test", "params:num_workers_loader"
                ],
                outputs="ncad_dataloader",
                name="ncad_dataloader_node"
            ),
            # checkpoint node
            node(
                func=set_up_callbacks,
                inputs=["params:model_dir", "params:log_dir"],
                outputs=["checkpointer", "logger", "exp_name"],
                name="callbacks_node"
            ),
            # NCAD node
            node(
                func=set_and_train_model,
                inputs=[
                    "ncad_dataloader",
                    "params:model_dir", 
                    "logger", 
                    "params:epochs", 
                    "params:limit_val_batches", 
                    "params:num_sanity_val_steps", 
                    "params:check_val_every_n_epoch", 
                    "checkpointer",
                    "ts_channels",
                    "params:window_length",
                    "params:suspect_window_length", 
                    "params:tcn_kernel_size",
                    "params:tcn_layers",
                    "params:tcn_out_channels",
                    "params:tcn_maxpool_out_channels",
                    "params:embedding_rep_dim",
                    "params:normalize_embedding",
                    # hpars for classifier
                    "params:distance",
                    "params:classifier_threshold",
                    "params:threshold_grid_length_test",
                    # hpars for optimizer
                    "params:learning_rate",
                    # hpars for anomalizers
                    "params:coe_rate",
                    "params:mixup_rate",
                    # hpars for validation and test
                    "params:stride_roll_pred_val_test",
                    "params:val_labels_adj",
                    "params:test_labels_adj",
                    "params:max_windows_unfold_batch",
                    "params:evaluation_result_path"
                ],
                outputs=["model_dir_linked", "classifier_threshold"],
                name="set_and_train_node"
            ),
            # evaluation node
            node(
                func=load_and_evaluate,
                inputs=["model_dir_linked", "exp_name", "test_std", "params:stride_roll_pred_val_test", "classifier_threshold"], 
                outputs=["iot_test_inference_scores", "iot_test_inference_votes"],
                name="evaluation_node"
            )
        ]
    )
