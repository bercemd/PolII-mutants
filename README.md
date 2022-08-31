# Prediction of phenotypes of PolII-mutants

This repository includes the scripts that use machine learning to train and predict phenotypes from sequence and molecular dynamics (MD) simulations data. All scripts are written in Python and uses TensorFlow. 

The input data files for RNA polymerase II trigger loop mutations are provided for the sequence-based and MD-based models. The trained models were provided for the predictive sequence-based model and Variational-Auto-Encoder (VAE) MD-based and fitness-based models.  

*** Requirements:
```
Python versions 3+
TensorFlow version 2.4+
Numpy version 1.19.5
Sklearn version 0.24.2
```
*** Usage
```
Python [options] filename.py
```
*** Examples:

Training the sequence data against the phenotypes:
```
python ml_sequence_training.py --optimizer=Adam --batch_size=100 --learning_rate=0.00001 --kl_weight=1.0 --model_dir=model --train_number=100 --test_number=35 --train_epochs=1000 --trainfile=train.seq.csv --testfile=test.seq.csv
```
Prediction of the phenotypes from amino acid sequence using a trained model:
```
python ml_sequence_prediction.py --model_dir=model --test_number=589 --input_weights=weights.sequence.best.hdf5 --testfile=prop.seq.all.csv --output_prd_test=prd.all.dat
```
Prediction of the phenotypes from fitness scores using a trained model:
```
python ml_fitness_prediction.py --model_dir=model --input_dim=63 --testfile=fitness.csv --input_weights=weights.fitness.prediction.h5 --output_prd=prd_phenotype.dat
```
Calculation of latent space coordinates from fitness scores using the trained variational autoencoder model:
```
python ml_fitness_vae.py --model_dir=model --sample_number=589 --input_dim=63 --testfile=fitness.csv --input_weights=weights.fitness.vae.h5 --output_latent=latent.fitness.dat
```
Training a variational autoencoder model using MD simulation data as the input:
```
python ml_md_vae_training.py --optimizer=Adam --learning_rate=0.0001 --model_dir=model --sample_number=135 --input_dim=62 --train_epochs=500 --trainfile=prop.md.csv
```
Calculation of latent space coordinates from MD simulation data using the trained variational autoencoder model:
```
python ml_md_vae_trained.py --model_dir=model --sample_number=135 --input_dim=62 --testfile=prop.md.csv --input_weights=weights.md.vae.h5 --output_latent=latent.md.dat
```
*** Citation
```
Bercem Dutagaci, Bingbing Duan, Chenxi Qiu, Craig D. Kaplan, Michael Feig, Characterization of RNA Polymerase II Trigger Loop Mutations using Molecular Dynamics Simulations and Machine Learning, 2022, bioRxiv, doi: https://doi.org/10.1101/2022.08.11.503690 
```
