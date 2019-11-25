# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 18:55:42 2019

@author: user
"""
#step 1 :load tweet_DM.json and translate into raw set that only have 'tweet_id' and 'text' column
#step 2 :according to 'data_identification.csv' spilt raw set into train set and test set
#step 3 :merge 'emotion.csv' and train set on 'tweet_id',then it's time to do some pre-processing
import pandas as pd
df = pd.read_csv('trainset.csv',lineterminator = '\n')
df.columns = ['id','text','emotion']
df['emotion']=df['emotion'].apply(lambda x:x.replace('\r',''))


#remove @blahblah ,numbers,hashtag and <LH>
import re
import numpy as np
def remove_pattern(input_txt,pattern):
    r = re.findall(pattern,input_txt)
    for i in r:
        input_txt = re.sub(i,' ',input_txt)
    return input_txt
df['text']=np.vectorize(remove_pattern)(df['text'],'@[\w]*')
df['text']=np.vectorize(remove_pattern)(df['text'],'<LH>')
df['text']=np.vectorize(remove_pattern)(df['text'],'#')
df['text']=np.vectorize(remove_pattern)(df['text'],'[0-9]')
df[0:5]
#tokenization,use tweet tokenizer
from nltk.tokenize import TweetTokenizer
token = TweetTokenizer(reduce_len = True)
df['text']=df['text'].str.lower()
df['text']=df['text'].apply(lambda x :token.tokenize(x))
df[0:5]
#remove stopwords and punctuation 
from nltk.corpus import stopwords
import string
stop = stopwords.words('english')
df['text']=df['text'].apply(lambda x :[item for item in x if item not in stop])
punctuation = list(string.punctuation)
punctuation.append('...')
df['text']=df['text'].apply(lambda x :[item for item in x if item not in punctuation])
df[0:5]
#stemming
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
df['text']=df['text'].apply(lambda x:[stemmer.stem(i) for i in x])
df[0:5]
#stitch tokens back together
df['text']=df['text'].apply(lambda x:' '.join(x))
df[0:5]



