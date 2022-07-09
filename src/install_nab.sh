cd anomaly_detection_spatial_temporal_data/model
git clone https://github.com/numenta/NAB.git
cd NAB
conda env remove -n NAB
conda env create 
rm -fr data/* results/*
source activate NAB
pip install juliacall==0.4.3
