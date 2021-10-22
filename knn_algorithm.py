from collections import Counter
import numpy as np

def euclideanDistance(data1, data2):
    sq_distance = 0
    for i in range(len(data1)):
        sq_distance += np.square(data1[i] - data2[i])
    return np.sqrt(sq_distance)


def knn(data, query, k):
    distances = []
    
    # Calculating euclidean distance between each row of training data and query
    for index, example in enumerate(data):
        distance = euclideanDistance(example[:-1], query)
        distances.append((distance, index))
    
    # Sorting them on the basis of distance
    sortdist = sorted(distances)
    
    # Pick the first k entries from the sorted collection
    neighbours = sortdist[:k]
    
    return neighbours
