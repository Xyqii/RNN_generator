# RNN_generator

## Introduction
Apply recurrent neural networks to molecular generation.

## Requirements:
* Python 2.7
* Tensorflow 1.5.0
* numpy 1.14.1

## Datasets
Corresponding SMILES sequences are provided in the four directories respectively according to different purposes.
We used [SMILES enumeration](https://github.com/Ebjerrum/SMILES-enumeration) to prepare the sequences.
The sequences are then converted to tokens during data prepocessing. The code used for tokenization is based on the [smiles_tokenizer](https://github.com/topazape/LSTM_Chem/blob/master/lstm_chem/utils/smiles_tokenizer.py) module in LSTM_Chem.  
Before preliminary training, the sequences can be preprocessed by running:
```
python pre_data.py
```
After preparing the preliminary data, the sequences used for transfer learning can be preprocessed by running:
```
python pre_data_tl.py
```

## Basic use
To train the preliminary model:
```
python model.py
```
To perform transfer learning:
```
python tl.py
```
To generate SMILES sequences:
```
python generate.py
```

## Experiments
The SMILES sequences were generated at random, so all the generated sequences were deposited in the four directories according to different purposes.
