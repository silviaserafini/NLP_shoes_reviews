import tensorflow as tf
from utils import *
import nltk
import pickle
import tensorflow_datasets as tfds
from numpy import dot
from numpy.linalg import norm
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import json
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('averaged_perceptron_tagger')



total_shoes_df=pd.read_pickle("raw_total_shoes_df.pkl")
total_shoes_df['content']=total_shoes_df['content'].apply(text_clean)
total_shoes_df['label'] = total_shoes_df['score'].apply(assign_label)
total_shoes_df['encoded_labels']=total_shoes_df['label'].apply(encode_labels)

#test and train set
texts=list(total_shoes_df['content'])
DIM_TRAIN=round(len(texts)*0.8)
dataset1=np.array(texts)

y=total_shoes_df['encoded_labels']
y=np.array(y)
y.reshape(13497,1)

df=pd.DataFrame({'text':texts,'labels':y})
datasetR = tf.data.Dataset.from_tensor_slices(df['text'].values)
datasetL = tf.data.Dataset.from_tensor_slices(df['labels'].values)
dataset = tf.data.Dataset.zip((datasetR, datasetL))
train_dataset, test_dataset = dataset.take(DIM_TRAIN), dataset.skip(DIM_TRAIN)

#batch
train_dataset = train_dataset.padded_batch(64)
test_dataset = test_dataset.padded_batch(64)

cos_sim = lambda a, b: dot(a, b)/(norm(a)*norm(b))

#loadin the model
embed = hub.load("https://tfhub.dev/google/nnlm-en-dim128/2")

model = tf.keras.Sequential([
  hub.KerasLayer('https://tfhub.dev/google/nnlm-en-dim128/2', trainable=True, input_shape=[], dtype=tf.string),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(1, activation='relu'),
])

#compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')

#training the model
model.fit(train_dataset , validation_data = test_dataset, validation_steps=30, epochs=10)

'''#saving the model
model1=model
moment=time.localtime()

name='review_classifier_TFH.h5'

model1_json = model1.to_json()
with open(name + '.json', "w") as json_file:
    json_file.write(model1_json)

# serialize weights to HDF5
model1.save_weights(name)
print("Saved model1 to disk")'''


