import gensim
import re
import json
import numpy as np
from cnnlstm import *
from keras.models import Sequential
from keras.layers import *
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.callbacks import TensorBoard
from keras.utils.np_utils import to_categorical
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
MAX_NEWS = 100000  # 用多少条新闻数据
MAX_NB_WORDS = 25000  # 保留单词数
MAX_LEN = 10  # 统一句长
EMBEDDING_DIM = 300  # word_embedding向量长度
one_cat_max = 10000

r = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, embedding_matrix = preprocess()

    # set parameters:
    batch_size = 200
    lstm_index = 100  # 100
    hidden_dims = 500  # 全连接层节点数300
    epochs = 15

    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    print('x_train shape:', X_train.shape)
    print('x_test shape:', X_test.shape)

    print('Build model...')
    model = Sequential()
    model.add(Embedding(embedding_matrix.shape[0],
                        EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=MAX_LEN,
                        trainable=False))

    model.add(LSTM(lstm_index))

    model.add(Dense(hidden_dims, activation='relu'))

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(rate=0.1))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(rate=0.1))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(len(y_train[0]), activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
   
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, y_test),
              verbose=2
              )

    print("RESULT ON TEST: ", model.evaluate(X_test, y_test, batch_size=100))
    '''
    for i in range(200):
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=30,
              validation_data=(X_test, y_test),
              verbose=2,
              callbacks=[tbCallBack])
    model.save(str(i) + 'model.h5')

    '''
