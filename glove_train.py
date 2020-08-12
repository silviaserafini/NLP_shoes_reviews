from keras.preprocessing import sequence
from utils import *
import tensorflow as tf
import nltk
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

total_shoes_df=pd.read_pickle("raw_total_shoes_df.pkl")
texts=list(total_shoes_df['content'])

def tokenize(text):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    MAX_SEQUENCE_LENGTH = 50
    padded_sequences = sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    word_index = tokenizer.word_index
    return padded_sequences, word_index

tokenized, word_index = tokenize(texts)

X=tokenized

total_shoes_df['label'] = total_shoes_df['score'].apply(assign_label)
y=total_shoes_df['label'].apply(encode_labels)

#test train split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, random_state=42)
print('X_test_dim',len(X_test),'Y_test_dim',len(Y_test),
      'X_train_dim',len(X_train),'Y_train_dim',len(Y_train))

VAL=round(0.8*len(X_test))

X_val = X_test[VAL:]
Y_val = Y_test[VAL:]

X_test = X_test[0:VAL]
Y_test = Y_test[0:VAL]

#GloVe
embeddings_index = {}
f = open('glove.6B/glove.6B.50d.txt', 'r', encoding = 'utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

#building of the embedding matrix
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = MAX_SEQUENCE_LENGTH

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


#LSTM model building

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

batch_size = 128

model = Sequential()

model.add(embedding_layer)

model.add(LSTM(64))

model.add(Dropout(0.50))

model.add(Dense(1, activation='sigmoid'))

# Training the LSTM model
# try using different optimizers and different optimizer configs

model.compile('adam', 'binary_crossentropy', metrics=['accuracy', precision, recall])

print('Train...')

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=16,
          validation_data=[X_test, Y_test])

x = model.evaluate(X_val, Y_val)

print("Loss: ", x[0])
print("Accuracy: ", x[1])
print("Precision: ", x[2])
print("Recall: ", x[3])

'''#saving the model
moment=time.localtime()
name='review_classifier.h5'

model_json = model.to_json()
with open(name + '.json', "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights(name)
print("Saved model to disk")'''