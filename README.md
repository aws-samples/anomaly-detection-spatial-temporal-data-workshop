# KDD 2022 Hands-on Tutorial: Anomaly Detection For Spatial Temporal Data

## Overview
This github repo is prepared for KDD 2022 hands-on tutorial. The project pipelines are prepared using the templates from [Kedro](https://kedro.readthedocs.io/en/stable/) 0.18.0. Kedro is an open-source Python framework for creating reproducible, maintainable and modular data science code. It borrows concepts from software engineering and applies them to machine-learning code; applied concepts include modularity, separation of concerns and versioning.  
"Kedro is a development workflow framework which aims to become the industry standard for developing production-ready code. Kedro helps structure your data pipeline using software engineering principles, eliminating project delays due to code rewrites and thereby providing more time to focus on building robust pipelines. Additionally, the framework provides a standardised approach to collaboration for teams building robust, scalable, deployable, reproducible and versioned data pipelines." --QuantumBlack, a McKinsey company


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

## References 
[1] Subutai Ahmad, Alexander Lavin, Scott Purdy, and Zuha Agha. 2017. Unsuper-
vised real-time anomaly detection for streaming data. Neurocomputing 262 (Nov.
2017), 134–147. https://doi.org/10.1016/j.neucom.2017.04.070
[2] Anisa Allahdadi, Ricardo Morla, and Jaime S. Cardoso. 2018. 802.11 Wireless
Simulation and Anomaly Detection using HMM and UBM. http://arxiv.org/abs/
1707.02933 Number: arXiv:1707.02933 arXiv:1707.02933 [cs].
[3] Jason Baumgartner, Savvas Zannettou, Brian Keegan, Megan Squire, and Jeremy
Blackburn. 2020. The Pushshift Reddit Dataset. http://arxiv.org/abs/2001.08435
Number: arXiv:2001.08435 arXiv:2001.08435 [cs].
[4] Chris U. Carmona, François-Xavier Aubet, Valentin Flunkert, and Jan Gasthaus.
2021. Neural Contextual Anomaly Detection for Time Series. http://arxiv.org/
abs/2107.07702 Number: arXiv:2107.07702arXiv:2107.07702 [cs].
[5] Jiho Choi, Taewook Ko, Younhyuk Choi, Hyungho Byun, and Chong-kwon
Kim. 2021. Dynamic graph convolutional networks with attention mechanism
for rumor detection on social media. PLOS ONE 16, 8 (Aug. 2021), e0256039.
https://doi.org/10.1371/journal.pone.0256039
[6] Yuwei Cui, Chetan Surpur, Subutai Ahmad, and Jeff Hawkins. 2016. A com-
parative study of HTM and other neural network models for online sequence
learning with streaming data. In 2016 International Joint Conference on Neural
Networks (IJCNN). IEEE, Vancouver, BC, Canada, 1530–1538. https://doi.org/10.
1109/IJCNN.2016.7727380
[7] Ailin Deng and Bryan Hooi. 2021. Graph Neural Network-Based Anomaly
Detection in Multivariate Time Series. http://arxiv.org/abs/2106.06947 Number:
arXiv:2106.06947 arXiv:2106.06947 [cs].
[8] Alexander Lavin and Subutai Ahmad. 2015. Evaluating Real-Time Anomaly
Detection Algorithms – The Numenta Anomaly Benchmark. In 2015 IEEE 14th
International Conference on Machine Learning and Applications (ICMLA). IEEE,
Miami, FL, USA, 38–44. https://doi.org/10.1109/ICMLA.2015.141
[9] Yixin Liu, Shirui Pan, Yu Guang Wang, Fei Xiong, Liang Wang, Qingfeng Chen,
and Vincent CS Lee. 2015. Anomaly Detection in Dynamic Graphs via Trans-
former. 14, 8 (2015), 13.
[10] Edgar Alonso Lopez-Rojas and Stefan Axelsson. 2014. BANKSIM: A BANK
PAYMENTS SIMULATOR FOR FRAUD DETECTION RESEARCH. (2014), 9.
[11] Martin Happ, Matthias Herlich, Christian Maier, Jia Lei Du, and Peter Dorfin-
ger. 2021. Graph-neural-network-based delay estimation for communication
networks with heterogeneous scheduling policies. ITU Journal on Future and
Evolving Technologies 2, 4 (June 2021), 1–8. https://doi.org/10.52953/TEJX5530
[12] José Suárez-Varela, Miquel Ferriol-Galmés, Albert López, Paul Almasan,
Guillermo Bernárdez, David Pujol-Perich, Krzysztof Rusek, Loïck Bonniot,
Christoph Neumann, François Schnitzler, François Taïani, Martin Happ, Chris-
tian Maier, Jia Lei Du, Matthias Herlich, Peter Dorfinger, Nick Vincent Hainke,
Stefan Venz, Johannes Wegener, Henrike Wissing, Bo Wu, Shihan Xiao, Pere
Barlet-Ros, and Albert Cabellos-Aparicio. 2021. The Graph Neural Network-
ing Challenge: A Worldwide Competition for Education in AI/ML for Net-
works. ACM SIGCOMM Computer Communication Review 51, 3 (July 2021),
9–16. https://doi.org/10.1145/3477482.3477485 arXiv:2107.12433 [cs].
[13] Riccardo Taormina, Stefano Galelli, Nils Ole Tippenhauer, Elad Salomons, Avi
Ostfeld, Demetrios G. Eliades, Mohsen Aghashahi, Raanju Sundararajan, Mohsen
Pourahmadi, M. Katherine Banks, B. M. Brentan, M. Herrera, Amin Rasekh,
Enrique Campbell, I. Montalvo, G. Lima, J. Izquierdo, Kelsey Haddad, Nikolaos
Gatsis, Ahmad Taha, Saravanakumar Lakshmanan Somasundaram, D. Ayala-
Cabrera, Sarin E. Chandy, Bruce Campbell, Pratim Biswas, Cynthia S. Lo, D.
Manzi, E. Luvizotto, Jr, Zachary A. Barker, Marcio Giacomoni, M. Fayzul K.
Pasha, M. Ehsan Shafiee, Ahmed A. Abokifa, Mashor Housh, Bijay Kc, and Ziv
Ohar. 2018. The Battle Of The Attack Detection Algorithms: Disclosing Cyber
Attacks On Water Distribution Networks. Journal of Water Resources Planning
and Management 144, 8 (Aug. 2018), 04018048. https://doi.org/10.1061/(ASCE)
WR.1943-5452.0000969
[14] Shen Wang and Philip S. Yu. 2022. Graph Neural Networks in Anomaly Detection.
In Graph Neural Networks: Foundations, Frontiers, and Applications, Lingfei Wu,
Peng Cui, Jian Pei, and Liang Zhao (Eds.). Springer Singapore, Singapore, 557–578.
https://doi.org/10.1007/978-981-16-6054-2_26
[15] Yulei Wu, Hong-Ning Dai, and Haina Tang. 2021. Graph Neural Networks for
Anomaly Detection in Industrial Internet of Things. IEEE Internet of Things
Journal (2021), 1–1. https://doi.org/10.1109/JIOT.2021.3094295
[16] Tong Zhao, Bo Ni, Wenhao Yu, Zhichun Guo, Neil Shah, and Meng Jiang.
2021. Action Sequence Augmentation for Early Graph-based Anomaly De-
tection. In Proceedings of the 30th ACM International Conference on Information
& Knowledge Management. 2668–2678. https://doi.org/10.1145/3459637.3482313
arXiv:2010.10016 [cs].
[17] Li Zheng, Zhenpeng Li, Jian Li, Zhao Li, and Jun Gao. 2019. AddGraph: Anomaly
Detection in Dynamic Graph Using Attention-based Temporal GCN. In Proceed-
ings of the Twenty-Eighth International Joint Conference on Artificial Intelligence.
International Joint Conferences on Artificial Intelligence Organization, Macao,
China, 4419–4425. https://doi.org/10.24963/ijcai.2019/614

![image](https://code.solutions-lab.ml.aws.dev/storage/user/63/files/ff1b2200-0362-11ed-900d-466cc63a8b32)

