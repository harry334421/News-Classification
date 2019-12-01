import keras
from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


(x_train,y_train),(x_test,y_test) = reuters.load_data(num_words=None,test_split=0.2)
word_index = reuters.get_word_index()

#print('number of training samples:{}'.format(len(x_train)))
#print('number of test samples:{}'.format(len(x_test)))

num_classes = max(y_train) + 1
#print('number of classes:{}'.format(num_classes))

#print(x_train[0])
#print(y_train[0])

index_to_word={}
for key, value in word_index.items():
    index_to_word[value] = key
#print(index_to_word[43])

#print(' '.join([index_to_word[x] for x in x_train[7]]))

max_words = 10000
tokenizer = Tokenizer(num_words = max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
'''
print(x_train.shape)
print(x_train[0])
print(y_train.shape)
print(y_train[0])
'''
#搭建网络
model = Sequential()
model.add(Dense(512, input_shape=(max_words,))) #添加512节点的全连接
model.add(Activation('relu')) #激活函数,negative to 0
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 32
epochs = 2

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,validation_split=0.1)
score = model.evaluate(x_test,y_test, batch_size=batch_size, verbose=1)

print('Test loss:{}'.format(score[0]))
print('Test accuracy:{}'.format(score[1]))

#Test loss:0.827447063670239
#Test accuracy:0.8063223361968994
#print(len(x_test)*0.8063223361968994) output: 1810.999967098236
