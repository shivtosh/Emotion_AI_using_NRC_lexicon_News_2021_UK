#!/usr/bin/env python
# coding: utf-8

# In[22]:


#import pandas as pd
import nltk
from nrclex import NRCLex
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string
import os, re, csv, json, sys, string
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

import gzip

from tqdm import tqdm

import pickle as pkl
from argparse import ArgumentParser
import logging


# In[ ]:


#still checking extension


# In[3]:


print(string.punctuation)


# In[4]:


def text_preprocessing(text):
    text=text.lower()
    text_p = "".join([char for char in text if char not in string.punctuation])
    text=text_p.strip()
    text=text.replace('“', '').replace('”', '')
    text=text.replace('’', '').replace('’', '')
    new_text = ''.join((x for x in text if not x.isdigit()))
    print(new_text)
    words = word_tokenize(new_text)
    stop_words = stopwords.words('english')
    filtered_words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in filtered_words])
    return(lemmatized_output)
    
    
    
    
    
    


# In[49]:


stop_words = stopwords.words('english')
with open("/Users/shivangidubey/Downloads/stopwords.txt", "w") as fobj:
    for x in stop_words:
        fobj.write(x + "\n")


# In[1]:


data=pd.read_csv("/Users/shivangidubey/Downloads/data_31_dec_to_25_mar.csv",index_col=0)
data.news_text=data.news_text.astype(str)
    


# In[2]:


for i in data.iterrows():
    #print(i[1][0])
    
    i[1][0]=text_preprocessing(i[1][0])


# In[34]:


data.iloc[0,0]


# In[36]:


data.rename(columns = {'news_text':'text'}, inplace = True)


# In[37]:


data['speaker']='UK'


# In[38]:


data.head()


# In[39]:


data.to_csv("/Users/shivangidubey/Downloads/jan_to_mar_data")


# In[40]:


def read_lexicon(path, LEXNAMES):
    df = pd.read_csv(path)
    df = df[~df['word'].isna()]
    df = df[['word']+LEXNAMES]
    df['word'] = [x.lower() for x in df['word']]
    return df


# In[41]:


def prep_dim_lexicon(df, dim):
    ldf = df[['word']+[dim]]
    ldf = ldf[~ldf[dim].isna()]
    ldf.drop_duplicates(subset=['word'], keep='first', inplace=True)
    ldf[dim] = [float(x) for x in ldf[dim]]
    ldf.rename({dim: 'val'}, axis='columns', inplace=True)
    ldf.set_index('word', inplace=True)
    return ldf


# In[42]:


def get_alpha(token):
    return token.isalpha()


# In[43]:


def get_vals(twt, lexdf):
    tt = twt.lower().split(" ")
    at = [w for w in tt if w.isalpha()]

    pw = [x for x in tt if x in lexdf.index]
    pv = [lexdf.loc[w]['val'] for w in pw]

    numTokens = len(at)
    numLexTokens = len(pw)
    
    avgLexVal = np.mean(pv)  #nan for 0 tokens

    return [numTokens, numLexTokens, avgLexVal]


# In[44]:



def process_df(df, lexdf):
    logging.info("Number of rows: " + str(len(df)))

    resrows = [get_vals(x, lexdf) for x in df['text']]
    resrows = [x + y for x,y in zip(df.values.tolist(), resrows)]

    resdf = pd.DataFrame(resrows, columns=df.columns.tolist() + ['numTokens', 'numLexTokens', 'avgLexVal'])
    resdf = resdf[resdf['numLexTokens']>=1]
    
    resdf['lexRatio'] = resdf['numLexTokens']/resdf['numTokens']
    return resdf


# In[45]:


def main(dataPath, LEXICON, LEXNAMES, savePath):

    os.makedirs(savePath, exist_ok=True)

    logfile = os.path.join(savePath, 'log.txt')

    logging.basicConfig(filename=logfile, format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
    
    df = pd.read_csv(dataPath)
    df.text=df.text.astype(str)
    

    for LEXNAME in LEXNAMES:

        lexdf = prep_dim_lexicon(LEXICON, LEXNAME)
        logging.info(LEXNAME + " lexicon length: " + str(len(lexdf)))
        resdf = process_df(df, lexdf)
    
        resdf.to_csv(os.path.join(savePath, LEXNAME+'.csv'), index=False)


# In[46]:


if __name__=='__main__':
    

    dataPath = "/Users/shivangidubey/Downloads/jan_to_mar_data"
    lexPath = "/Users/shivangidubey/Downloads/lexicon.csv"

    LEXNAMES = ["Positive","Negative","Anger","Anticipation","Disgust","Fear","Joy","Sadness","Surprise","Trust"]
    LEXICON = read_lexicon(lexPath, LEXNAMES)

    savePath = "/Users/shivangidubey/Downloads/emo_dyn.csv"

    main(dataPath, LEXICON, LEXNAMES, savePath)


# In[59]:


anger=pd.read_csv("/Users/shivangidubey/Downloads/emo_dyn.csv/Anger.csv",index_col=0)


# In[60]:


anger.head()


# In[ ]:




