import gensim
import re
import json
import pickle
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import *
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.callbacks import TensorBoard
from keras.utils.np_utils import to_categorical

MAX_NEWS = 60000  # 用多少条新闻数据
MAX_NB_WORDS = 20000  # 保留单词数
MAX_LEN = 16  # 统一句长
EMBEDDING_DIM = 300  # word_embedding向量长度
one_cat_max = 4000  # 一种类型在数据集中的最大值

r = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"


def get_labels(file_dir):
    file = open(file_dir, 'r', encoding='utf-8')
    cates = []
    for line in file.readlines():
        dic = json.loads(line)
        cates.append(dic['category'])

    category_list = list(set(cates))

    with open('labels.pickle', 'wb') as save_dir:
        pickle.dump(category_list, save_dir)


def readfile(path):
    with open(path, 'rb') as file:
        re = pickle.load(file)
    return re


def preprocess():
    file = open("G:/Python/News_Category_Dataset_v2.json", 'r', encoding='utf-8')
    headlines = []
    cates = []
    count = 0
    politics = 0
    entertain = 0

    for line in file.readlines():
        dic = json.loads(line)
        if (dic['category'] == 'POLITICS'):  # 控制政治类新闻不要太多
            politics += 1
            if (politics >= one_cat_max):
                continue

        if (dic['category'] == 'ENTERTAINMENT'):  # 控制政治类新闻不要太多
            entertain += 1
            if (entertain >= one_cat_max):
                continue

        headline = text_to_word_sequence(dic['headline'],
                                         filters=r,
                                         lower=True,
                                         split=" ")
        headlines.append(headline)
        cates.append(dic['category'])
        count += 1

        if count >= MAX_NEWS:
            break

    category_list = readfile(
        'G:/Python/LSTM_Class/bert-NewsClassifier/labels.pickle')  # 加载用所有数据整理的标题集合，共41个
    category_to_id = {category_list[i]: i for i in range(len(category_list))}

    labels = [category_to_id[cates[i]] for i in range(len(cates))]

    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(headlines)
    sequences = tokenizer.texts_to_sequences(headlines)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_LEN)

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    data = np.array(data)
    labels = np.array(labels)
    print("shape info:", data.shape, '\n', labels.shape)

    X_train, X_test, y_train, y_test = train_test_split(data,
                                                        labels,
                                                        test_size=0.2,
                                                        shuffle=True)

    # 导入WORD2VEC模型
    w2v = gensim.models.KeyedVectors.load_word2vec_format('G:/Python/GoogleNews-vectors-negative300.bin.gz', binary=True)

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if word in w2v:
            embedding_vector = w2v[word]
        else:
            embedding_vector = np.random.uniform(-0.001, 0.001, 300)  # 绝对值小的随机向量表示未登录词

        embedding_matrix[i] = embedding_vector

    return X_train, X_test, y_train, y_test, embedding_matrix


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, embedding_matrix = preprocess()

    # set parameters:
    batch_size = 250
    filters = 250  # CNN 250
    kernel_size = 3  # CNN 3
    poolsize = 2  # maxpooling
    lstm_index = 100  # 100
    hidden_dims = 300  # 全连接层节点数300
    epochs = 20

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
                        trainable=True))
    model.add(SpatialDropout1D(0.5))

    model.add(Conv1D(filters=filters,
                     kernel_size=kernel_size,
                     padding='valid',
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=poolsize))

    model.add(LSTM(lstm_index, dropout=0.5, recurrent_dropout=0.5))

    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(rate=0.5))

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(rate=0.5))

    model.add(Dense(hidden_dims, activation='relu'))


    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(len(y_train[0]), activation='softmax'))  # 41是文本类型数

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    '''

    model=load_model('train20000')
    '''

    '''
    tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                             histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
                             batch_size=200,  # 用多大量的数据计算直方图
                             write_graph=True,  # 是否存储网络结构图
                             write_grads=True,  # 是否可视化梯度直方图
                             write_images=True,  # 是否可视化参数
                             )
                             '''
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, y_test),
              verbose=2,
              # callbacks=[tbCallBack]
              )

    '''

    aa = model.predict(X_test)  # 具体看测试集预测情况
    for i in range(300):
        print(aa[i].argmax(), y_test[i].argmax(), '\n')
    '''

    print("RESULT ON TEST: ", model.evaluate(X_test, y_test, batch_size=200))

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
