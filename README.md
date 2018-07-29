# battery-aging
Auto Encoder for anomaly detection on batteries

**************************
For train and evaluate use 
python run.py train
python run.py evaluate

**************************
Synthetic Data Toolchain:
**************************
### PYTHON ###
python Mercurio.py export # export battery voltage and current as CSV episodes for matlab
### MATLAB + SIMULINK ###
Generate synthetic data with generate_all_synthetic_data([100,95,90,85,80])
### PYTHON ###
Convert back as dataframe with original timestamp the synthetic data with
python Mercurio.py import
**************************

