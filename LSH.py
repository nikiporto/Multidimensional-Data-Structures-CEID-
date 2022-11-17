import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(r'MulDimDataStr/news.csv',engine='python')
df=df.drop(columns=['author', 'date','headlines','read_more','ctext'])
#print(df)
#print('dimension: ', df.shape)
df.head()

from sklearn.feature_extraction.text import TfidfVectorizer
#------------TF-IDF

tfidf = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 3),
    min_df=0,
    stop_words='english')
X_tfidf = tfidf.fit_transform(df['text'])
#print("TF-IDF :")
#print(X_tfidf)


#---------CREATION OF BUCKETS
#The idea behind LSH is to generate random lines/vectors. 
# Then everything that falls below this line has a negative score and will fall into what we'll be referring to as bin 0.

def random_vectors(dim, n_vectors):
    """
    generate random projection vectors
    """
    return np.random.randn(dim, n_vectors)


#--------generating LSH lsh_model
from collections import defaultdict


def LSH(X_tfidf, n_vectors):    
    
    dim = X_tfidf.shape[1]
    rand_vectors = random_vectors(dim, n_vectors)  

    # partition data points into bins,
    # and encode bin index bits into integers
    bit_bin_pointers = X_tfidf.dot(rand_vectors) >= 0
    powers_of_two = 1 << np.arange(n_vectors - 1, -1, step=-1)  # x << y is the same as multiplying x by 2 ** y --> shift to the left by y places
    bin_pointers = bit_bin_pointers.dot(powers_of_two)

    # update `table` so that `table[i]` is the list of document ids with bin index equal to i
    table = defaultdict(list)
    for idx, bin_index in enumerate(bin_pointers):
        table[bin_index].append(idx)
    
    model = {'table': table,
             'random_vectors': rand_vectors,
             'bin_pointers': bin_pointers,
             'bit_bin_pointers': bit_bin_pointers}
    return model


#--------train the lsh_model
n_vectors = 11
lsh_model = LSH(X_tfidf, n_vectors)


#----------Querying using the LSH model
from itertools import combinations


def search_near_bins(query_bin_bits, table, search_radius=3, candidate_set=None):
    """
    For a given query vector and trained LSH model's table
    return all candidate neighbors with the specified search radius.
    
    """
    if candidate_set is None:
        candidate_set = set()

    n_vectors = query_bin_bits.shape[0]
    powers_of_two = 1 << np.arange(n_vectors - 1, -1, step=-1)

    for different_bits in combinations(range(n_vectors), search_radius):
        # flip the bits of the query bin to produce a new bit vector
        index = list(different_bits)
        alternate_bits = query_bin_bits.copy()
        alternate_bits[index] = np.logical_not(alternate_bits[index])

        # convert the new bit vector to an integer index
        nearby_bin = alternate_bits.dot(powers_of_two)

        # fetch the list of documents belonging to
        # the bin indexed by the new bit vector,
        # then add those documents to candidate_set;
        if nearby_bin in table:
            candidate_set.update(table[nearby_bin])

    return candidate_set


# we use search_near_bins logic to search
# for similar documents and return a dataframe 
# that contains the most similar data points 
# according to LSH and their corresponding distances.

from sklearn.metrics.pairwise import pairwise_distances


def nearest_neighbour(X_tfidf, query_vector, lsh_model, max_search_radius):
    table = lsh_model['table']
    random_vectors = lsh_model['random_vectors']

    # compute bin index for the query vector, in bit representation.
    bin_index_bits = np.ravel(query_vector.dot(random_vectors) >= 0)

    # search nearby bins and collect candidates
    candidate_set = set()
    for search_radius in range(max_search_radius + 1):
        candidate_set = search_near_bins(bin_index_bits, table, search_radius, candidate_set)
    #print(candidate_set)
    # sort candidates by their true distances from the query
    candidate_list = list(candidate_set)
    candidates = X_tfidf[candidate_list]
    distance = pairwise_distances(candidates, query_vector, metric='cosine').flatten()
    
    distance_col = 'distance'
    nearest_neighbors = pd.DataFrame({
        'id': candidate_list, distance_col: distance
    }).sort_values(distance_col).reset_index(drop=True)
    
    return nearest_neighbors


import time

item_id = 19
query_vector = X_tfidf[item_id]
start_time = time.time()
nearest_neighbors = nearest_neighbour(X_tfidf, query_vector, lsh_model, max_search_radius=4)
end_time = time.time()
print(nearest_neighbors)


total=end_time-start_time

print("time duration of LSH:",total)
