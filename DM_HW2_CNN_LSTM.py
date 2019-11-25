# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 04:56:05 2019

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:28:47 2019

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:45:58 2019

@author: user
"""

import pandas as pd
import keras
import numpy as np

###load training,testing set,dealwith null value(due to pre-processing)
data = pd.read_csv('cleantrain_set.csv')
testdata = pd.read_csv('cleantest_set.csv')
data.dropna(inplace = True)
testdata.fillna('thisisnull',inplace=True)
### Create sequence
from keras.preprocessing.text import Tokenizer
#take most often 5000 words to word_dic
vocabulary_size = 5000 
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(data['text'])
#transform words to a sequence
sequences = tokenizer.texts_to_sequences(data['text'])
###check length of sequence
import matplotlib.pyplot as plt
text_len = [len(x) for x in sequences]
histo = pd.Series(text_len).value_counts()
histo.sort_index(inplace=True)
histo = pd.DataFrame(histo,columns = ['counts'])
histo[0:25].plot.bar(figsize=(8,6))
plt.xticks(rotation=360)
plt.show()
#pad or cut each sequence to proper length
sequence_len = int(8) 
from keras.preprocessing.sequence import pad_sequences
train = pad_sequences(sequences, maxlen=sequence_len)
###Label encoding
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(data['emotion'])
print('check label: ', le.classes_)
print('\n## Before convert')
print('data[emotion][0:4]:\n', data['emotion'][0:4])
print('\n data[emotion].shape: ', data['emotion'].shape)
#transform sentumental label to one-hot label
def label_encode(le, labels):
    enc = le.transform(labels)
    return keras.utils.to_categorical(enc)
def label_decode(le, one_hot_label):
    dec = np.argmax(one_hot_label, axis=1)
    return le.inverse_transform(dec)
data_label = label_encode(le, data['emotion'])
print('\n\n## After convert')
print('data_label[0:4]:\n', data_label[0:4])
print('\n data_label.shape: ', data_label.shape)

###split train,valid set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, data_label, 
                                                    test_size=0.25,shuffle = True,random_state = 17)

###deep learning
#input is padded sequence,output is one-hot label
input_shape = X_train.shape[1]
print('input_shape: ', input_shape)
output_shape = len(le.classes_)
print('output_shape: ', output_shape)
from keras.models import Model
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import LSTM,Activation,concatenate,Input
from keras.models import load_model
from keras import optimizers 
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D,MaxPooling1D
from keras.callbacks import CSVLogger,EarlyStopping,ModelCheckpoint
csv_logger = CSVLogger('training_log.csv')
#use GPU to accelerate training
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#hyper parameters
learning_rate = 0.001
epochs = 50
batch_size = 512
#build CNN-LSTM model
model_input = Input(shape = (input_shape,))
embed = Embedding(output_dim=32,
                    input_dim=vocabulary_size+1, 
                    input_length=input_shape)(model_input)
BN1 = BatchNormalization()(embed)
cnn2=Conv1D(filters=64, kernel_size=2,padding='same',activation='relu')(BN1)                     
MP2 = MaxPooling1D(pool_size=2)(cnn2)
cnn3=Conv1D(filters=64, kernel_size=3,padding='same',activation='relu')(BN1)                
MP3 = MaxPooling1D(pool_size=2)(cnn3)
cnnn=Conv1D(filters=64, kernel_size=6,padding='same',activation='relu')(BN1)                    
MPn = MaxPooling1D(pool_size=1)(cnnn)
cnn_out = concatenate([MP2,MP3,MPn], axis=-2)
LSTM1 = LSTM(units=128,unroll = True, recurrent_initializer='orthogonal',
             return_sequences=False,
               dropout=0.2, recurrent_dropout=0.2)(cnn_out)
LSTM1_BN = BatchNormalization()(LSTM1)
FC1 = Dense(units=512)(LSTM1_BN)
FC1_BN = BatchNormalization()(FC1)
Activate  = Activation('relu')(FC1_BN)
Drop = Dropout(0.5)(Activate)
FC2 = Dense(units=output_shape,activation='softmax' )(Drop)
model = Model(inputs=model_input, outputs=FC2)
print(model.summary())
# define loss function & optimizer
adam = optimizers.Adam(lr=learning_rate)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#callback csv_logger,Modelcheckpoint and Earlystopping
earlystopping=EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
BEST_MODEL_DIR = "C:/Users/user/Downloads/DMHW2/best_lstm.h5"
save_best = ModelCheckpoint(BEST_MODEL_DIR, monitor='val_loss', verbose=0, 
                            save_best_only=True, save_weights_only=False, mode='auto', period=1)
#train the model
history = model.fit(X_train, y_train, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    callbacks=[csv_logger,earlystopping,save_best],
                    validation_data = (X_test, y_test))
print('training finish')



###predict on public test set and creat submission file
#load the model
SAVE_MODEL_DIR = "C:/Users/user/Downloads/DMHW2/best_lstm.h5"
model = load_model(SAVE_MODEL_DIR)
#tokenize test data and generate the sequence 
test_sequences = tokenizer.texts_to_sequences(testdata['text'])
_testdata = pad_sequences(test_sequences, maxlen=sequence_len)
pred_result = model.predict(_testdata, batch_size=512)
pred_result = label_decode(le, pred_result)
pred = pd.Series(pred_result)
testdata['emotion']=pred
testdata.drop(['text'],axis = 1,inplace = True)
testdata.to_csv('108011557_3CNN64_LSTM128_em32_batch512_voc5000.csv',index = 0)
###confusion matrix

#from sklearn.metrics import confusion_matrix
#y_test_pred = model.predict(X_test, batch_size=1024)
#
#y_test_pred = label_decode(le, y_test_pred)
#y_test = label_decode(le, y_test)
#matrix = confusion_matrix(y_test, y_test_pred)
#matrixdf = pd.DataFrame(matrix,index=le.classes_,columns =le.classes_)
#print('confusion matrix = \n',matrixdf)
