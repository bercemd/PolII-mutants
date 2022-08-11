# Prediction of phenotypes of PolII-mutants

This repository includes the scripts that use machine learning to train and predict phenotypes from sequence and molecular dynamics (MD) simulations data. All scripts are written in Python and uses TensorFlow. 

The input data files for RNA polymerase II trigger loop mutations are provided for the sequence-based and MD-based models. The trained models were provided for the predictive sequence-based model and Variational-Auto-Encoder (VAE) MD-based and fitness-based models.  

*** Requirements:

Python versions 3+
TensorFlow version 2.4+
Numpy version 1.19.5
Sklearn version 0.24.2

*** Usage

Python [options] filename.py

*** Examples:

Training the sequence data against the phenotypes:

python ml_sequence_training.py --optimizer=Adam --batch_size=100 --learning_rate=0.00001 --kl_weight=1.0 --model_dir=model --train_number=100 --test_number=35 --train_epochs=1000 --trainfile=train.seq.csv --testfile=test.seq.csv

Prediction of the phenotypes using a trained model:

python ml_sequence_prediction.py --model_dir=model --test_number=589 --input_weights=weights.sequence.best.hdf5 --testfile=prop.seq.all.csv --output_prd_test=prd.all.dat

*** Citation

Bercem Dutagaci, Bingbing Duan, Chenxi Qiu, Craig D. Kaplan, Michael Feig, Characterization of RNA Polymerase II Trigger Loop Mutations using Molecular Dynamics Simulations and Machine Learning, 2022