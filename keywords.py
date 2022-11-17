#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import numpy as np



# In[2]:


#read csv and keep the columns we need


# In[3]:


df=pd.read_csv(r'MulDimDataStr/news.csv',engine='python')


# In[4]:


news=df.drop(columns=['author', 'date','headlines','read_more','ctext'])


# In[5]:


#Find keywords


# In[6]:


from nltk import tokenize
from operator import itemgetter
import math


# In[7]:


def check_sent(word, sentences): 
    final = [all([w in x for w in word]) for x in sentences] 
    sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
    return int(len(sent_len))


# In[8]:


def get_top_n(dict_elem, n):
    result = dict(sorted(dict_elem.items(), key = itemgetter(1), reverse = True)[:n]) 
    return result


# In[9]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
stop_words = set(stopwords.words('english'))


# In[10]:


for i in range(4514):  
   doc=news.text[i] 
   total_words = doc.split()
   total_word_length = len(total_words)
   #print(total_word_length)
   total_sentences = tokenize.sent_tokenize(doc)
   total_sent_len = len(total_sentences)
   #print(total_sent_len)
   tf_score = {}
   for each_word in total_words:
       each_word = each_word.replace('.','')
       if each_word not in stop_words:
           if each_word in tf_score:
                   tf_score[each_word] += 1
           else:
               tf_score[each_word] = 1

   # Dividing by total_word_length for each dictionary element
   tf_score.update((x, y/int(total_word_length)) for x, y in tf_score.items())
   #print(tf_score) 
   idf_score = {}
   for each_word in total_words:
       each_word = each_word.replace('.','')
       if each_word not in stop_words:
           if each_word in idf_score:
               idf_score[each_word] = check_sent(each_word, total_sentences)
           else:
               idf_score[each_word] = 1

   # Performing a log and divide
   idf_score.update((x, math.log(int(total_sent_len)/y)) for x, y in idf_score.items())

   #print(idf_score)
   tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}
   #print(tf_idf_score)
   mydict=(get_top_n(tf_idf_score, 2))
   news.text[i]=list(mydict.keys())


# In[11]:


news


# # Making the points with wordembeddings
# 

# In[18]:


from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot


# In[13]:


words=[]
for i in news.text:
    for k in i:
        words.append(k)


# In[15]:


len(words)


# In[35]:


dataforw2v=news['text'].to_numpy()
dataforw2v


# In[14]:


words=np.unique(words)


# In[40]:


w2v = Word2Vec(dataforw2v, min_count=1,vector_size = 1)








# In[59]:


pointsdot=[]
for  i in dataforw2v:
        pointsdot.append(w2v.wv[i].T)
    #print(pointsdot[0:10])

points = []
for sublist in pointsdot:
        for item in sublist:
            points.append(item)
print(points[0:10])    

df = pd.DataFrame(points, columns = ['cord1', 'cord2'])

import os  
os.makedirs('MulDimDataStr', exist_ok=True)  
df.to_csv('MulDimDataStr/out2d.csv')  
# In[60]:











