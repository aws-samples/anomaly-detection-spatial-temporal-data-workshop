# KDD 2022 Hands-on Tutorial: Anomaly Detection For Spatial Temporal Data

## Overview
This github repo is prepared for KDD 2022 hands-on tutorial. The project pipelines are prepared using the templates from Kedro 0.18.0. 


## Setting up the environment

We declared dependencies for different pipelines for different use cases and prepared shell script to install the virtual environment. Once the virtual environment is installed, you can run the notebook using the customized env/kernel. Also, user can run the corresponding pipeline after activating the virtual env. 
For example, to run the financial fraud detection pipeline using the TADDY(dynamic graph based) modeling framework, follow these steps below: 
1. Prepare the Kedro Taddy virtual environment 
Run the following command:
```
```

2. Activate the virtual environment
Run the following command:
```
```

3. Run the pipeline 
Note that kedro pipeline has to be initiated from the repo root directory. So run the following command: 
```
```

[insert a kedro pipeline visualization here]

## Data Summary
We found and used different datasets for different use cases for this hands-on tutorial to cover enough variations in raw data format and structure. We illustrated different ways to convert the raw data to intermediate data that can be consumed in different modeling framework.  

## Model Summary 

## Mind Map 

## Run the pipelines 

This is your new Kedro project, which was generated using `Kedro 0.18.0`.

Take a look at the [Kedro documentation](https://kedro.readthedocs.io) to get started.

### Instructions on running Kedro pipeline 

You can run the entire pipeline for one use case with the corresponding activated virtual environment:

```
kedro run
```
You can also run your specific Kedro pipeline(sub-pipeline) with:

```
kedro run --pipeline <pipeline_name_in_registry>
```
You can even run your specific Kedro node function in the pipeline(sub-pipeline) with:

```
kedro run --node <node_name_in_registry>
```
For more details, you can run the command:
```
kedro run -h
```
#### For financial fraud use case 


#### For IoT network anomaly use case 


#### For Wifi network anomaly use case


#### For Reddit user behavior use case
### Instructions on running notebooks
You can select the custom kernel after installing the corresponding virtual environment for each use case. 
#### For financial fraud use case 


#### For IoT network anomaly use case 


#### For Wifi network anomaly use case


#### For Reddit user behavior use case

## Outline of the Tutorial



