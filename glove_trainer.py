from keras.preprocessing import sequence
from utils import assign_label, encode_labels, precision, recall
import tensorflow as tf
#import nltk
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

class Glove_trainer:
    def __init__(self,data_to_load_path, model_save_path, glove_path):
        self.EMBEDDING_DIM = 50
        self.BATCH_SIZE = 128
        self.MAX_SEQUENCE_LENGTH = 50
        self.total_shoes_df = pd.read_pickle(data_to_load_path)
        self.texts = list(self.total_shoes_df['content'])
        self.glove_path = glove_path
        self.embeddings_index = {}
        self.embedding_matrix = []
        self.model = Sequential()
        self.padded_sequences = []
        self.word_index = {}
        self.model_save_path = model_save_path
        self.data_to_load_path = data_to_load_path
    
    def train(self):
        self.tokenize()
        self.load_glove_dictionary()
        self.create_embeddings_matrix()
        self.build_LSTM_model()
        self.train_LSTM_model()

    def tokenize(self):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.texts)
        sequences = tokenizer.texts_to_sequences(self.texts)
        self.padded_sequences = sequence.pad_sequences(sequences, maxlen = self.MAX_SEQUENCE_LENGTH)
        self.word_index = tokenizer.word_index

    def test_train_split(self):
        self.total_shoes_df['label'] = self.total_shoes_df['score'].apply(assign_label)
        y = self.total_shoes_df['label'].apply(encode_labels)
        X = self.padded_sequences
        #test train split
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, random_state=42)
        VAL=round(0.8*len(X_test))
        print('X_test_dim',len(X_test),'Y_test_dim',len(Y_test),
      'X_train_dim',len(X_train),'Y_train_dim',len(Y_train))
        X_val = X_test[VAL:]
        Y_val = Y_test[VAL:]
        X_test = X_test[0:VAL]
        Y_test = Y_test[0:VAL]
        print('VAL=',VAL,'X_val_dim',len(X_val),'Y_val_dim',len(Y_val),'X_test_dim',len(X_test),'Y_test_dim',len(Y_test),
      'X_train_dim',len(X_train),'Y_train_dim',len(Y_train))
        return X_train, Y_train, X_test, Y_test, X_val, Y_val

    def load_glove_dictionary(self):
        print('loading glove dictionary')
        f = open(self.glove_path, 'r', encoding = 'utf-8')
        #GloVe 
        print('creating word vectors')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()
        print('Found %s word vectors.' % len(self.embeddings_index))
    
    def create_embeddings_matrix(self):
        print('creating embedding matrix...')  
        self.embedding_matrix = np.zeros((len(self.word_index) + 1, self.EMBEDDING_DIM))
        for word, i in self.word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector

    def build_LSTM_model(self):
        print('model building...')  
        embedding_layer = Embedding(len(self.word_index) + 1,
                            self.EMBEDDING_DIM,
                            weights=[self.embedding_matrix],
                            input_length=self.MAX_SEQUENCE_LENGTH,
                            trainable=False)  
        self.model.add(embedding_layer)
        self.model.add(LSTM(64))
        self.model.add(Dropout(0.50))
        self.model.add(Dense(1, activation = 'sigmoid'))
                
    def train_LSTM_model(self):
        print('model compiling...')       
        self.model.compile('adam', 'binary_crossentropy', metrics=['accuracy', precision, recall])
        print('model train...')
        X_train,Y_train, X_test, Y_test, X_val, Y_val = self.test_train_split()
        self.model.fit(X_train, Y_train,
                batch_size=self.BATCH_SIZE,
                epochs=16,
                validation_data=[X_test, Y_test])
        x = model.evaluate(X_val, Y_val)
        print("Loss: ", x[0])
        print("Accuracy: ", x[1])
        print("Precision: ", x[2])
        print("Recall: ", x[3])
        self.save()

    def save(self):
        moment = time.localtime()
        model_json = self.model.to_json()
        with open( self.model_save_path, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(self.model_save_path[:-5])
        print("Saved model to disk")
