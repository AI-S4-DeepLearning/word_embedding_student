import numpy as np
import math
import pickle


def vector_hash(vec, planes):
    """ Hash a vector given the dividing planes. """
    h = vec @ planes >= 0
    # h = np.squeeze(h)
    
    hash_value = 0
    for i in range(h.size):
        hash_value += 2**i * h[i]
        
    return hash_value


def make_hash_table(vecs: np.array, n_buckets):
    """ Create a locality-sensitive hash table with `n_buckets` buckets
        of given vectors. """
    # Buckets should be positive and a multiple of 2
    n_buckets = max(1, n_buckets)
    n_planes = math.ceil(np.log2(n_buckets))
    n_buckets = 2**n_planes

    # Create the planes randomly
    n_dims = vecs.shape[1]
    planes = np.random.normal(size=(n_dims, n_planes))

    print(f"Creating a locality-sensitive hash table with {n_buckets} buckets.")
    buckets = {index: [] for index in range(n_buckets)}   # hash -> vec
    lookup = {index: [] for index in range(n_buckets)}    # hash -> id
    
    # Vecs is a list of vector embeddings
    for i, vec in enumerate(vecs):
        h = vector_hash(vec, planes)
        buckets[h].append(vec)
        lookup[h].append(i)
        
    return buckets, lookup, planes

