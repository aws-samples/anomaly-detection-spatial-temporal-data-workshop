Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

SPDX-License-Identifier: Apache-2.0

# Pipeline

## Overview

This repo contains multiple pipelines registered in `pipeline_registry.py`. 

## Pipeline Summary

### Data Pipelines 
|   Pipeline Name                  |     Purpose        |
| -----------------------------    | ------------------------------------------------------- |
| financial_data_processing        | transform raw transaction data into graph for TADDY     |
| financial_data_processing_to_ts  | transform raw transaction data into time series for NAB |
| iot_gdn_data_processing          | transform raw sensor data into formats for GDN          |
| iot_nab_data_processing          | transform raw sensor data into time series for NAB      |
| iot_ncad_data_processing         | transform raw sensor data into graphs for NCAD          |
| reddit_data_processing           | transform raw Reddit data into graph for ELAND          |
| wifi_gdn_data_processing         | transform raw wifi nework data into formats for GDN     |

### Model Training/Inference Pipelines 

|   Pipeline Name                  |     Purpose        |
| --------------------------       |  -------------------------------------------------------|
| nab_model                        | train NAB model on time series                          |
| ncad                             | train NCAD model on time series                         |
| gdn                              | train GDN model on multivariate time series             |
| taddy                            | train TADDY model on financial transaction network data |
| eland                            | train ELAND model on Reddit Posts/Comments graph        |


# Model 

This folder contains class and functions for NAB, NCAD, ELAND, TADDY and GDN modeling framework 

# Feature

This folder contains class and functions to extract node features for the user behavior use case. 

# Data

This folder contains class and functions to transfrom raw data to intermediate data formats. 

