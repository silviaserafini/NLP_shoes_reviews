import tensorflow as tf
from utils import text_clean, assign_label, encode_labels
import nltk
import pickle
from numpy import dot
from numpy.linalg import norm
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import json
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout

class TSH_traier:

  def __init__(self,data_to_load_path, model_save_path):
    self.data_to_load_path = data_to_load_path
    self.model_save_path = model_save_path
    self.total_shoes_df = pd.read_pickle(self.data_to_load_path)
    self.texts = []
    self.model = []

  def train(self):
    nltk.download('averaged_perceptron_tagger')
    self.build_model()
    self.compile_model()
    self.train_model()

  def test_train_split(self):

    cos_sim = lambda a, b: dot(a, b)/(norm(a)*norm(b))
    self.total_shoes_df['content'] = self.total_shoes_df['content'].apply(text_clean)
    self.texts = list(self.total_shoes_df['content'])
    DIM_TRAIN=round(len(self.texts)*0.8)
    self.total_shoes_df['label'] = self.total_shoes_df['score'].apply(assign_label)
    self.total_shoes_df['encoded_labels'] = self.total_shoes_df['label'].apply(encode_labels)
    #test and train set
    
    y = np.array(self.total_shoes_df['encoded_labels'])
    y = y.reshape(13497,)
    print('y_len=',y.shape,'texts_len=',len(self.texts))
    df = pd.DataFrame({'text': self.texts,'labels':y})
    datasetR = tf.data.Dataset.from_tensor_slices(df['text'].values)
    datasetL = tf.data.Dataset.from_tensor_slices(df['labels'].values)
    dataset = tf.data.Dataset.zip((datasetR, datasetL))
    train_dataset, test_dataset = dataset.take(DIM_TRAIN), dataset.skip(DIM_TRAIN)
    #batch
    train_dataset = train_dataset.padded_batch(64)
    test_dataset = test_dataset.padded_batch(64)
    return train_dataset, test_dataset

  def build_model(self):
      print('building the model...')
      #loadin the model
      #embed = hub.load("https://tfhub.dev/google/nnlm-en-dim128/2")
      self.model = tf.keras.Sequential([
        hub.KerasLayer('https://tfhub.dev/google/nnlm-en-dim128/2', trainable=True, input_shape=[], dtype=tf.string),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='relu'),
      ])

  def compile_model(self):
    print('compiling the model...')
    self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')

  def train_model(self):
    print('training the model...')
    train_dataset, test_dataset = self.test_train_split()
    self.model.fit(train_dataset , validation_data = test_dataset, validation_steps=30, epochs=10)
    self.save_model()

  def save_model(self):
      print('saving the model...')
      moment = time.localtime()
      model_json = self.model.to_json()
      with open( self.model_save_path, "w") as json_file:
          json_file.write(model_json)
      # serialize weights to HDF5
      model.save_weights(self.model_save_path[:-5])
      print("Saved model to disk")
      print('done!:-)')


