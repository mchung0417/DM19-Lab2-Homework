# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:22:47 2019

@author: user
"""

import pandas as pd

### training data
anger_train = pd.read_csv("data/semeval/train/anger-ratings-0to1.train.txt",
                         sep="\t", header=None,names=["id", "text", "emotion", "intensity"])
sadness_train = pd.read_csv("data/semeval/train/sadness-ratings-0to1.train.txt",
                         sep="\t", header=None, names=["id", "text", "emotion", "intensity"])
fear_train = pd.read_csv("data/semeval/train/fear-ratings-0to1.train.txt",
                         sep="\t", header=None, names=["id", "text", "emotion", "intensity"])
joy_train = pd.read_csv("data/semeval/train/joy-ratings-0to1.train.txt",
                         sep="\t", header=None, names=["id", "text", "emotion", "intensity"])
# combine 4 sub-dataset
train_df = pd.concat([anger_train, fear_train, joy_train, sadness_train], ignore_index=True)
### testing data
anger_test = pd.read_csv("data/semeval/dev/anger-ratings-0to1.dev.gold.txt",
                         sep="\t", header=None, names=["id", "text", "emotion", "intensity"])
sadness_test = pd.read_csv("data/semeval/dev/sadness-ratings-0to1.dev.gold.txt",
                         sep="\t", header=None, names=["id", "text", "emotion", "intensity"])
fear_test = pd.read_csv("data/semeval/dev/fear-ratings-0to1.dev.gold.txt",
                         sep="\t", header=None, names=["id", "text", "emotion", "intensity"])
joy_test = pd.read_csv("data/semeval/dev/joy-ratings-0to1.dev.gold.txt",
                         sep="\t", header=None, names=["id", "text", "emotion", "intensity"])

# combine 4 sub-dataset
test_df = pd.concat([anger_test, fear_test, joy_test, sadness_test], ignore_index=True)
train_df.head()
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
vectorizer = CountVectorizer()
train_vec = vectorizer.fit_transform(train_df.text)
train_word = vectorizer.get_feature_names()

words = {}
for i in range(len(train_word)):
    words[i]=train_word[i]
tf = pd.DataFrame(train_vec.toarray())
tf = tf.rename(words,axis = 1)
tf.loc['frequency']=tf.sum(axis = 0)

tf_= pd.DataFrame(tf.loc['frequency'])
tf_.sort_values(by = 'frequency',ascending=False,inplace = True)
tf_ = tf_.apply(lambda x :x/tf_['frequency'].sum())

tf_[0:30].plot.bar(figsize = (12,10))
plt.xticks(rotation=45)
plt.title('train_df word frequency')
plt.show()
test_vec = vectorizer.fit_transform(test_df.text)
test_word = vectorizer.get_feature_names()

test_words = {}
for i in range(len(test_word)):
    test_words[i]=test_word[i]
test_tf = pd.DataFrame(test_vec.toarray())
test_tf = test_tf.rename(test_words,axis = 1)
test_tf.loc['frequency']=test_tf.sum(axis = 0)

test_tf_= pd.DataFrame(test_tf.loc['frequency'])
test_tf_.sort_values(by = 'frequency',ascending=False,inplace = True)
test_tf_ = test_tf_.apply(lambda x :x/test_tf_['frequency'].sum())

test_tf_[0:30].plot.bar(figsize = (12,10))
plt.xticks(rotation=45)
plt.title('test_df word frequency')
plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
Tfidf_vectorizer = TfidfVectorizer(max_features=1000,tokenizer = nltk.word_tokenize,use_idf=True)
train_tfidf_vec = Tfidf_vectorizer.fit_transform(train_df.text)
train_tfidf_word = Tfidf_vectorizer.get_feature_names()
train_tfidf_vec.shape
train_tfidf_vec.toarray()
train_tfidf_word[100:110]
import nltk

# build analyzers (bag-of-words)
BOW_500 = CountVectorizer(max_features=500, tokenizer=nltk.word_tokenize) 

# apply analyzer to training data
BOW_500.fit(train_df['text'])

train_data_BOW_features_500 = BOW_500.transform(train_df['text'])

from sklearn.tree import DecisionTreeClassifier

# for a classificaiton problem, you need to provide both training & testing data
X_train = BOW_500.transform(train_df['text'])
y_train = train_df['emotion']

X_test = BOW_500.transform(test_df['text'])
y_test = test_df['emotion']

## take a look at data dimension is a good habbit  :)
print('X_train.shape: ', X_train.shape)
print('y_train.shape: ', y_train.shape)
print('X_test.shape: ', X_test.shape)
print('y_test.shape: ', y_test.shape)
## build DecisionTree model
DT_model = DecisionTreeClassifier(random_state=0)

## training!
DT_model = DT_model.fit(X_train, y_train)

## predict!
y_train_pred = DT_model.predict(X_train)
y_test_pred = DT_model.predict(X_test)

## so we get the pred result
y_test_pred[:10]
## accuracy
from sklearn.metrics import accuracy_score

acc_train = accuracy_score(y_true=y_train, y_pred=y_train_pred)
acc_test = accuracy_score(y_true=y_test, y_pred=y_test_pred)

print('training accuracy: {}'.format(round(acc_train, 2)))
print('testing accuracy: {}'.format(round(acc_test, 2)))
## precision, recall, f1-score,
from sklearn.metrics import classification_report

print(classification_report(y_true=y_test, y_pred=y_test_pred))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true=y_test, y_pred=y_test_pred) 
# Funciton for visualizing confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes, title='Confusion matrix',
                          cmap=sns.cubehelix_palette(as_cmap=True)):
    """
    This function is modified from: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    classes.sort()
    tick_marks = np.arange(len(classes))    
    
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels = classes,
           yticklabels = classes,
           title = title,
           xlabel = 'True label',
           ylabel = 'Predicted label')

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    ylim_top = len(classes) - 0.5
    plt.ylim([ylim_top, -.5])
    plt.tight_layout()
    plt.show()
    # plot your confusion matrix
my_tags = ['anger', 'fear', 'joy', 'sadness']
plot_confusion_matrix(cm, classes=my_tags, title='Confusion matrix')
#444
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train,y_train)
nb_y_test_pred = nb.predict(X_test)
print(classification_report(y_true=y_test, y_pred=nb_y_test_pred))
nb_cm = confusion_matrix(y_true=y_test, y_pred=nb_y_test_pred) 
plot_confusion_matrix(nb_cm, classes=my_tags, title='Confusion matrix')