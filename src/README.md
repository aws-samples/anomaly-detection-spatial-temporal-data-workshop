Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

SPDX-License-Identifier: Apache-2.0

# `src` folder

## Setting up environments

This folder contains the setup scripts to prepare the environments for each of the five models. To prepare the environments, please run each of the the `prepare_<model>_environment.sh` scripts. The scripts will create a new python virtual environment, install the requirements in this environment, and register the environment with the Jupyter kernel.

```
# model is `eland`, `gdn`, `nab`, `ncad`, and `taddy`
chmod +x prepare_<model>_environment.sh 
./prepare_<model>_environment.sh 
```

## Core tutorial code as `kedro` pipelines

The core tutorial modules are defined in the `anomaly_detection_spatial_temporal_data` folder. The tutorial can be run through both Jupyter notebooks and kedro pipelines. 

To run the notebooks:
1. Navigate to `cd ../notebooks/`
2. Run the notebooks

To run the kedro pipelines:
1. Navigate to root: `cd ..`
2. Activate a specific virtual environment: `source src/kedro-<model>-venv`
3. Depending on the model (see mind-map), specify the dataset in `conf/base/parameters.yml`
