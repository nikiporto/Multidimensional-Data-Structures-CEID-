#!/usr/bin/env python
# coding: utf-8

# In[281]:


import pandas as pd
import re
import numpy as np


# # read csv and keep the columns we need
# 

# In[282]:


df=pd.read_csv(r'MulDimDataStr//news.csv',engine='python')


# In[283]:


news=df.drop(columns=['author', 'date','headlines','read_more','ctext'])


# # Find keywords

# In[284]:


from nltk import tokenize
from operator import itemgetter
import math


# In[285]:


def check_sent(word, sentences): 
    final = [all([w in x for w in word]) for x in sentences] 
    sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
    return int(len(sent_len))


# In[286]:


def get_top_n(dict_elem, n):
    result = dict(sorted(dict_elem.items(), key = itemgetter(1), reverse = True)[:n]) 
    return result


# In[287]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
stop_words = set(stopwords.words('english'))


# In[288]:


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
   mydict=(get_top_n(tf_idf_score, 3))
   news.text[i]=list(mydict.keys())


# # Making the points with wordembeddings
# 

# In[289]:


from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot


# In[290]:


words=[]
for i in news.text:
    for k in i:
        words.append(k)


# In[291]:


len(words)


# In[292]:


dataforw2v=news['text'].to_numpy()
dataforw2v


# In[293]:


words=np.unique(words)


# In[294]:


w2v = Word2Vec(dataforw2v, min_count=1, vector_size = 1)
print(w2v)


# In[295]:




# In[296]:


pointsdot=[]
for  i in dataforw2v:
        pointsdot.append(w2v.wv[i].T)
    #print(pointsdot[0:10])

points = []
for sublist in pointsdot:
        for item in sublist:
            points.append(item)

# In[297]:




# In[299]:


points


# # KD TREE 

# In[224]:


class KDTree(object):
    

    
    def __init__(self, points, dim, dist_sq_func=None):
      
        if dist_sq_func is None:
            dist_sq_func = lambda a, b: sum((x - b[i]) ** 2 
                for i, x in enumerate(a))
                
        def make(points, i=0):
            if len(points) > 1:
                points.sort(key=lambda x: x[i])
                i = (i + 1) % dim
                m = len(points) >> 1
                return [make(points[:m], i), make(points[m + 1:], i), 
                    points[m]]
            if len(points) == 1:
                return [None, None, points[0]]
        
        def add_point(node, point, i=0):
            if node is not None:
                dx = node[2][i] - point[i]
                for j, c in ((0, dx >= 0), (1, dx < 0)):
                    if c and node[j] is None:
                        node[j] = [None, None, point]
                    elif c:
                        add_point(node[j], point, (i + 1) % dim)

        
        
        
        
        
        import heapq
        def get_knn(node, point, k, return_dist_sq, heap, i=0, tiebreaker=1):
            if node is not None:
                dist_sq = dist_sq_func(point, node[2])
                dx = node[2][i] - point[i]
                if len(heap) < k:
                    heapq.heappush(heap, (-dist_sq, tiebreaker, node[2]))
                elif dist_sq < -heap[0][0]:
                    heapq.heappushpop(heap, (-dist_sq, tiebreaker, node[2]))
                i = (i + 1) % dim
                # Goes into the left branch, then the right branch if needed
                for b in (dx < 0, dx >= 0)[:1 + (dx * dx < -heap[0][0])]:
                    get_knn(node[b], point, k, return_dist_sq, 
                        heap, i, (tiebreaker << 1) | b)
            if tiebreaker == 1:
                return [(-h[0], h[2]) if return_dist_sq else h[2] 
                    for h in sorted(heap)][::-1]

        def walk(node):
            if node is not None:
                for j in 0, 1:
                    for x in walk(node[j]):
                        yield x
                yield node[2]

        self._add_point = add_point
        self._get_knn = get_knn 
        self._root = make(points)
        self._walk = walk

    def __iter__(self):
        return self._walk(self._root)
        
    def add_point(self, point):

        if self._root is None:
            self._root = [None, None, point]
        else:
            self._add_point(self._root, point)

    def get_knn(self, point, k, return_dist_sq=True):

        return self._get_knn(self._root, point, k, return_dist_sq, [])

    
  


# In[278]:


kd_tree_results = []



import random
import time

dim=3
#points = points[1:4299]
#additional_points =points[4300:4400]
#query_points = points[4401:4514]

global kd_tree
kd_tree = KDTree(points, dim)
#for point in additional_points:
#               kd_tree.add_point(point)

t0 = time.time()        
kd_tree_results.append(tuple(kd_tree.get_knn([ -0.03476865962147713] * dim, 5)))
t1 = time.time()


# In[279]:


print('Generation time: ',t1-t0)


# In[280]:


kd_tree_results


# In[ ]:




