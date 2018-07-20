from face_recognition.Graph import Graph
from face_recognition.Node import Node
import numpy as np

def computeDists(vectors, n):

    ''' 
    Computes the L2 distances between all elements of the vectors in the inpu
    
    input : vectors (M , 128) array vectors

    output : array of shape (M,M) of all the distances

    '''
    if type(vectors[0]) != np.ndarray:
        vectors = np.array(vectors)
    x = vectors.reshape(n, 128)
    y = vectors.reshape(n, 128)
    dists = -2 * np.matmul(x, y.T)
    dists +=  np.sum(x**2, axis=1)[:, np.newaxis]
    dists += np.sum(y**2, axis=1)

    return  np.sqrt(dists)

def createGraph(dists, threshold_similarity):
    '''
    Creates a graph of all the nodes (images) and edges(distances)

    input : 
    dists :  (M,M) array of distances
    threshold_similarity : the cutoff value for identifying similar and different faces

    output : 
    graph of nodes and edges

    '''
    graph = Graph()

    for i in range(dists.shape[0]):
        node = Node(i,i)
        graph.addNode(node)

    for ind in np.ndindex(dists.shape):
        if ind[0] != ind[1] and dists[ind] < threshold_similarity:
            graph.addEdge((1/dists[ind])**2, ind[0], ind[1])
    
    return graph

def finalizeLabels(graph):
    for i in range(1000):
        graph.update()
    return graph