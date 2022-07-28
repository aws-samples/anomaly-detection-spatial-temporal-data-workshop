Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

SPDX-License-Identifier: Apache-2.0

# Notebooks

## Pre-req

Please ensure the python virtual environments for the Eland, GDN, NAB, NCAD, and TADDY models have been set up. To do so, navigate to the `src/` folder and run the five `prepare_<model>_environment.sh`, where `<model>` is `eland`, `gdn`, `nab`, `ncad`, and `taddy`.

```
#navigate to src folder in root
cd ../src/

# model is `eland`, `gdn`, `nab`, `ncad`, and `taddy`
chmod +x prepare_<model>_environment.sh 
./prepare_<model>_environment.sh 
```

## Overview

This folder contains the tutorial material as Jupyter Notebooks for the four anomaly detection use-cases: financial fraud, industrial iot, telecom network, and user behavior. 

## Instructions

0. Run `download_data.ipynb` to download the data. For the financial fraud dataset, you will need to create a Kaggle account to download the data.

1. Run each of the notebooks in the four sub-folders for each use-case.