from numpy import *
from knn import *

class SpectralClustering:

    def __init__(self,X=None,k=5):
        self.X = X
        self.k = k
        self.c = 0.05

    def loadfile(self,filename):
        self.X = loadtxt(filename)[:,0:2]

    def distanceGraph(self,X):

        knn = KNN(X,k)
        n = shape(X)[0]
        W = zeros((n,n))

        for i in range(n):
            distances = knn.getNNDist(i);
            neigbors = knn.getNN(i);
            for j in range(k):
                W[neighbors(j),i]= W[i,neighbors(j)] = distances[j]

    def similarityMatrix(self,W):
        return exp(W/c)


