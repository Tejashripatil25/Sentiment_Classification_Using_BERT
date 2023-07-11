### Sentiment_Analysis_NLP

![image](https://github.com/Tejashripatil25/Sentiment_Classification_Using_BERT/assets/124791646/53204c1a-f7ef-41e0-a4e2-57d42d4d668a)

BERT stands for Bidirectional Representation for Transformers, was proposed by researchers at Google AI language in 2018.

The main aim of that was to improve the understanding of the meaning of queries related to Google Search, 

BERT becomes one of the most important and complete architecture for various natural language tasks having generated state-of-the-art results on Sentence pair classification task, question-answer task, etc.

### Architecture:

One of the most important features of BERT is that its adaptability to perform different NLP tasks with state-of-the-art accuracy (similar to the transfer learning we used in Computer vision). For that, the paper also proposed the architecture of different tasks.

We will be using BERT architecture for single sentence classification tasks specifically the architecture used for CoLA (Corpus of Linguistic Acceptability) binary classification task. 

![image](https://github.com/Tejashripatil25/Sentiment_Classification_Using_BERT/assets/124791646/a230d216-0548-4fbf-8954-f02529b65a88)

BERT has proposed in the two versions:

#### BERT (BASE): 12 layers of encoder stack with 12 bidirectional self-attention heads and 768 hidden units.

#### BERT (LARGE): 24 layers of encoder stack with 24 bidirectional self-attention heads and 1024 hidden units.

For TensorFlow implementation, Google has provided two versions of both the BERT BASE and BERT LARGE: Uncased and Cased. In an uncased version, letters are lowercased before WordPiece tokenization.

### Example:-

! git clone https://github.com / google-research / bert.git

### Download BERT BASE model from tF hub ! wget https://storage.googleapis.com / bert_models / 2018_10_18 / uncased_L-12_H-768_A-12.zip ! unzip uncased_L-12_H-768_A-12.zip

% tensorflow_version 1.x

we will import modules necessary for running this project, we will be using NumPy, scikit-learn and Keras from TensorFlow inbuilt modules. These are already preinstalled in colab, make sure to install these in your environment.

import os

import re

import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow import keras

import csv

from sklearn import metrics

we will load IMDB sentiments datasets and do some preprocessing before training. For loading the IMDB dataset from TensorFlow Hub

### load data from positive and negative directories and a columns that takes there\

### positive and negative label

def load_directory_data(directory):

data = {}

data["sentence"] = []

data["sentiment"] = []

for file_path in os.listdir(directory):

 with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
	
 data["sentence"].append(f.read())
	
 data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))

return pd.DataFrame.from_dict(data)

### Merge positive and negative examples, add a polarity column and shuffle.

def load_dataset(directory):

pos_df = load_directory_data(os.path.join(directory, "pos"))

neg_df = load_directory_data(os.path.join(directory, "neg"))

pos_df["polarity"] = 1

neg_df["polarity"] = 0

return pd.concat([pos_df, neg_df]).sample(frac = 1).reset_index(drop = True)

### Download and process the dataset files.

def download_and_load_datasets(force_download = False):

dataset = tf.keras.utils.get_file(

 fname ="aclImdb.tar.gz",
	
 origin ="http://ai.stanford.edu/~amaas / data / sentiment / aclImdb_v1.tar.gz",
	
 extract = True)

train_df = load_dataset(os.path.join(os.path.dirname(dataset),
									"aclImdb", "train"))

test_df = load_dataset(os.path.join(os.path.dirname(dataset),
									"aclImdb", "test"))

return train_df, test_df

train, test = download_and_load_datasets()

train.shape, test.shape

### sample 5k datapoints for both train and test

train = train.sample(5000)

test = test.sample(5000)

### List columns of train and test data

train.columns, test.columns

### code

### Convert training data into BERT format

train_bert = pd.DataFrame({

'guid': range(len(train)),

'label':train['polarity'],

'alpha': ['a']*train.shape[0],

'text': train['sentence'].replace(r'\n', '', regex = True)
})

train_bert.head()

print("-----")

### convert test data into bert format

bert_test = pd.DataFrame({

'id':range(len(test)),

'text': test['sentence'].replace(r'\n', ' ', regex = True)
})

bert_test.head()

### split data into train and validation set

bert_train, bert_val = train_test_split(train_bert, test_size = 0.1)

### save train, validation and testfile to afolder

bert_train.to_csv('bert / IMDB_dataset / train.tsv', sep ='\t', index = False, header = False)

bert_val.to_csv('bert / IMDB_dataset / dev.tsv', sep ='\t', index = False, header = False)

bert_test.to_csv('bert / IMDB_dataset / test.tsv', sep ='\t', index = False, header = True)

### Most of the arguments hereare self-explanatory but some arguments needs to be explained:

 task name:We have discussed this above .Here we need toperform binary classification that why we use cola

 vocab file : A vocab file (vocab.txt) to map WordPiece to word id.

 init checkpoint: A tensorflow checkpoint required. Here we used downloaded bert.

 max_seq_length :caps the maximunumber of words to each reviews

 bert_config_file: file contains hyperparameter settings ! python bert / run_classifier.py

--task_name = cola --do_train = true --do_eval = true

--data_dir =/content / bert / IMDB_dataset

--vocab_file =/content / uncased_L-12_H-768_A-12 / vocab.txt

--bert_config_file =/content / uncased_L-12_H-768_A-12 / bert_config.json

--init_checkpoint =/content / uncased_L-12_H-768_A-12 / bert_model.ckpt

--max_seq_length = 64

--train_batch_size = 8 --learning_rate = 2e-5

--num_train_epochs = 3.0

--output_dir =/content / bert_output/

--do_lower_case = True

--save_checkpoints_steps 10000
