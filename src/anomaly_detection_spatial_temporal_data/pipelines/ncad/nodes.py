"""
This is a boilerplate pipeline 'ncad'
generated using Kedro 0.18.1
"""

import os
import time
import json
from typing import List, Union, Optional, Tuple, Dict

from pathlib import Path, PosixPath

import pandas as pd
import numpy as np
import itertools

import re

import torch
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import ncad
from ncad.ts import TimeSeries, TimeSeriesDataset
from ncad.ts import transforms as tr
from ncad.model import NCAD, NCADDataModule

import tqdm

def load_dataset(train_clean: pd.DataFrame, train_anom: pd.DataFrame, test: pd.DataFrame, sensor_cols: List[str], label_col: str) -> Tuple:
    if type(sensor_cols) == str:
        sensor_cols = sensor_cols.split("\n")
    train_dataset = TimeSeriesDataset()
    test_dataset = TimeSeriesDataset()
    
    
    for sensor in sensor_cols:
        train_dataset.append(
            TimeSeries(
                values=train_clean[sensor].values,
                labels=None,
                item_id=f"{sensor}_train_clean"
            )
        )
        
        train_dataset.append(
            TimeSeries(
                values=train_anom[sensor].values,
                labels=train_anom[label_col].values,
                item_id=f"{sensor}_train_anom"
            )
        )
        
        test_dataset.append(
            TimeSeries(
                values=test[sensor].values,
                labels=test[label_col].values,
                item_id=f"{sensor}_test"
            )
        )
           
    return train_dataset, test_dataset

def standardize_dataset(train_set: TimeSeriesDataset, test_set: TimeSeriesDataset) -> Tuple:
    # Standardize TimeSeries values (substract median, divide by interquartile range)
    scaler = tr.TimeSeriesScaler(type="robust")
    train_set = TimeSeriesDataset(ncad.utils.take_n_cycle(scaler(train_set), len(train_set)))
    test_set = TimeSeriesDataset(ncad.utils.take_n_cycle(scaler(test_set), len(test_set)))
    ts_channels = train_set[0].shape[1]
    return train_set, test_set, ts_channels


def batadal_inject_anomalies(
    dataset: TimeSeriesDataset,
    injection_method: str,
    ratio_injected_spikes: float,
) -> TimeSeriesDataset:

    # without this, kedro treats the output as a list of multiple elements rather than
    scratch = ""
    # dataset is transformed using a TimeSeriesTransform depending on the type of injection
    if injection_method == "None":
        return dataset
    elif injection_method == "local_outliers":
        if ratio_injected_spikes is None:
            # Inject synthetic anomalies: LocalOutlier
            ts_transform = tr.LocalOutlier(area_radius=500, num_spikes=360)
        else:
            ts_transform = tr.LocalOutlier(
                area_radius=700,
                num_spikes=ratio_injected_spikes,
                spike_multiplier_range=(1.0, 3.0),
                # direction_options = ['increase'],
            )

        # Generate many examples of injected time series
        multiplier = 20
        ts_transform_iterator = ts_transform(itertools.cycle(dataset))
        dataset_transformed = ncad.utils.take_n_cycle(
            ts_transform_iterator, multiplier * len(dataset)
        )
        dataset_transformed = TimeSeriesDataset(dataset_transformed)
    else:
        raise ValueError(f"injection_method = {injection_method} not supported!")

    return dataset_transformed


def split_test_into_val(test_set: TimeSeriesDataset, window_length: int, suspect_window_length: int) -> Tuple:
    # Split test dataset in two, half for validation and half for test
    _, validation_set, test_set = ncad.ts.split_train_val_test(
        data=test_set,
        val_portion=0.5,  # No labels in training data that could serve for validation
        test_portion=0.5,  # This dataset provides test set, so we don't need to take from training data
        split_method="past_future_with_warmup",
        split_warmup_length=window_length - suspect_window_length,
        verbose=False,
    )
    
    return validation_set, test_set
    
    
def construct_data_module(
    train_set_transformed: TimeSeriesDataset, validation_set: TimeSeriesDataset, test_set: TimeSeriesDataset,
    window_length: int, suspect_window_length: int, 
    num_series_in_train_batch: int,
    num_crops_per_series: int,
    stride_roll_pred_val_test: int,
    num_workers_loader: int
    
) -> NCADDataModule:
    # Define DataModule for training with lighting (window cropping + window injection)#
    data_module = NCADDataModule(
        train_ts_dataset=train_set_transformed,
        validation_ts_dataset=validation_set,
        test_ts_dataset=test_set,
        window_length=window_length,
        suspect_window_length=suspect_window_length,
        num_series_in_train_batch=num_series_in_train_batch,
        num_crops_per_series=num_crops_per_series,
        label_reduction_method="any",
        stride_val_and_test=stride_roll_pred_val_test,
        num_workers=num_workers_loader,
    )
    
    return data_module

def set_and_train_model(
    data_module: NCADDataModule, 
    model_dir: Union[str, PosixPath],
    logger: TensorBoardLogger,
    epochs: int,
    limit_val_batches: float,
    num_sanity_val_steps: int,
    check_val_every_n_epoch: int,
    checkpoint_cb: ModelCheckpoint,
    ts_channels: int,
    window_length: int,
    suspect_window_length: int, 
    tcn_kernel_size: int,
    tcn_layers: int,
    tcn_out_channels: int,
    tcn_maxpool_out_channels: int,
    embedding_rep_dim: int,
    normalize_embedding: bool,
    # hpars for classifier
    distance: str,
    classifier_threshold: float,
    threshold_grid_length_test: float,
    # hpars for optimizer
    learning_rate: float,
    # hpars for anomalizers
    coe_rate: float,
    mixup_rate: float,
    # hpars for validation and test
    stride_roll_pred_val_test: int,
    val_labels_adj: bool,
    test_labels_adj: bool,
    max_windows_unfold_batch: Optional[int] = 5000,
    
    evaluation_result_path: Optional[Union[str, PosixPath]] = None,
) -> Tuple:
    if distance == "cosine":
        # For the contrastive approach, the cosine distance is used
        distance = ncad.model.distances.CosineDistance()
    elif distance == "L2":
        # For the contrastive approach, the L2 distance is used
        distance = ncad.model.distances.LpDistance(p=2)
    elif distance == "non-contrastive":
        # For the non-contrastive approach, the classifier is
        # a neural-net based on the embedding of the whole window
        distance = ncad.model.distances.BinaryOnX1(rep_dim=embedding_rep_dim, layers=1)

    # Instantiate model #
    model = NCAD(
        ts_channels=ts_channels,
        window_length=window_length,
        suspect_window_length=suspect_window_length,
        # hpars for encoder
        tcn_kernel_size=tcn_kernel_size,
        tcn_layers=tcn_layers,
        tcn_out_channels=tcn_out_channels,
        tcn_maxpool_out_channels=tcn_maxpool_out_channels,
        embedding_rep_dim=embedding_rep_dim,
        normalize_embedding=normalize_embedding,
        # hpars for classifier
        distance=distance,
        classification_loss=nn.BCELoss(),
        classifier_threshold=classifier_threshold,
        threshold_grid_length_test=threshold_grid_length_test,
        # hpars for anomalizers
        coe_rate=coe_rate,
        mixup_rate=mixup_rate,
        # hpars for validation and test
        stride_rolling_val_test=stride_roll_pred_val_test,
        val_labels_adj=val_labels_adj,
        test_labels_adj=test_labels_adj,
        max_windows_unfold_batch=max_windows_unfold_batch,
        # hpars for optimizer
        learning_rate=learning_rate,
    )
    
    trainer = Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        default_root_dir=model_dir,
        logger=logger,
        min_epochs=epochs,
        max_epochs=epochs,
        limit_val_batches=limit_val_batches,
        num_sanity_val_steps=1,
        check_val_every_n_epoch=check_val_every_n_epoch,
        callbacks=[checkpoint_cb],
        # callbacks=[checkpoint_cb, earlystop_cb, lr_logger],
        auto_lr_find=False,
    )
    
    trainer.fit(
        model=model,
        datamodule=data_module,
    )
    
    # Metrics on validation and test data #
    evaluation_result = trainer.test()
    evaluation_result = evaluation_result[0]
    classifier_threshold = evaluation_result["classifier_threshold"]
    
    # Save evaluation results
    if evaluation_result_path is not None:
        path = evaluation_result_path
        path = PosixPath(path).expanduser() if str(path).startswith("~") else Path(path)
        with open(path, "w") as f:
            json.dump(evaluation_result, f, cls=ncad.utils.NpEncoder)

    for key, value in evaluation_result.items():
        # if key.startswith('test_'):
        print(f"{key}={value}")
        
    return model_dir, classifier_threshold
    
def set_up_callbacks(
    ## General
    model_dir: Union[str, PosixPath],
    log_dir: Union[str, PosixPath],
) -> ModelCheckpoint:
    # Experiment name #
    time_now = time.strftime("%Y-%m-%d-%H%M%S", time.localtime())
    exp_name = f"batadal-{time_now}"

    ### Training the model ###

    logger = TensorBoardLogger(save_dir=log_dir, name=exp_name)

    # Checkpoint callback, monitoring 'val_f1'
    checkpoint_cb = ModelCheckpoint(
        monitor="val_f1",
        dirpath=model_dir,
        filename="ncad-model-" + exp_name + "-{epoch:02d}-{val_f1:.4f}",
        save_top_k=1,
        mode="max",
    )
    
    return checkpoint_cb, logger, exp_name
    
def load_and_evaluate(
    model_dir,
    exp_name: str,
    data_set: TimeSeriesDataset,
    stride: int,
    classifier_threshold: float
):
    model_dir = PosixPath(model_dir).expanduser() if str(model_dir).startswith("~") else Path(model_dir)
    
    # Load top performing checkpoint
    # ckpt_path = [x for x in model_dir.glob('*.ckpt')][-1]
    ckpt_file = [
        file
        for file in os.listdir(model_dir)
        if (file.endswith(".ckpt") and file.startswith("ncad-model-" + exp_name))
#         if (file.endswith(".ckpt")) # debug only
    ]
    print(model_dir, len(ckpt_file), ckpt_file)
    ckpt_file = ckpt_file[-1]
    ckpt_path = model_dir / ckpt_file
    model = NCAD.load_from_checkpoint(ckpt_path)

    anomaly_probs_avg, anomaly_vote = model.tsdetect(data_set, stride, classifier_threshold)
    
    print(anomaly_probs_avg.shape, anomaly_vote.shape)
    print(f"NCAD on batadal dataset finished successfully!")    
    
    return anomaly_probs_avg, anomaly_vote
