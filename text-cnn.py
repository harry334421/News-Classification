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

MAX_NEWS = 30000  # 用多少条新闻数据
MAX_NB_WORDS = 25000  # 保留单词数
MAX_LEN = 10  # 统一句长
EMBEDDING_DIM = 300  # word_embedding向量长度
one_cat_max = 3000  # 一种类型在数据集中的最大值

r = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, embedding_matrix = preprocess()

    # set parameters:
    batch_size = 200
    filters = 250  # CNN 250
    kernel_size = 3  # CNN 3
    poolsize = 2  # maxpooling
    hidden_dims = 500  # 全连接层节点数300
    epochs = 30

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

    model.add(Conv1D(filters=filters,
                     kernel_size=kernel_size,
                     padding='valid',
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=poolsize))
    model.add(Flatten())

    model.add(Dense(hidden_dims, activation='relu'))

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(rate=0.1))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(rate=0.1))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(rate=0.1))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(rate=0.1))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(len(y_train[0]), activation='softmax'))  # 41是文本类型数

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                             histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
                             batch_size=200,  # 用多大量的数据计算直方图
                             write_graph=True,  # 是否存储网络结构图
                             write_grads=True,  # 是否可视化梯度直方图
                             write_images=True,  # 是否可视化参数
                             )
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, y_test),
              verbose=2,
              callbacks=[tbCallBack]
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
