from numpy import *
from knn import *
import pylab as pl

class SpectralClustering:

    def __init__(self,X=None):
        self.X = X
        self.k = 5
        self.c = 0.05

    def loadfile(self,filename):
        self.X = loadtxt(filename)[:,0:2]
        self.y = loadtxt(filename)[:,2]

    def distanceGraph(self,X):

        knn = KNN(X,self.k)
        n = shape(X)[0]
        W = zeros((n,n))

        for i in range(n):
            distances = knn.getNNDist(i);
            neighbors = knn.getNN(i);
            for j in range(self.k):
                W[neighbors[j],i]= W[i,neighbors[j]] = distances[j]

        return W

    def similarityMatrix(self,W):
        return exp(W/self.c)

    def latentVariables(self,X,simMat):

        degreeMat = diag(sum(simMat,axis=1).squeeze().tolist())

        U,D,V = linalg.svd(degreeMat - simMat)

        last2 = V[:,-2:]

        return last2

    def cluster(self,k=5,c=0.05):

        self.k = k
        self.c = c

        self.distGraph = self.distanceGraph(self.X)
        self.simMat = self.similarityMatrix(self.distGraph)
        self.latent = self.latentVariables(self.X,self.simMat)

        
        label0 = mat([self.latent[i,:].tolist() for i in range(shape(self.X)[0]) if self.y[i] == 0])
        label1 = mat([self.latent[i,:].tolist() for i in range(shape(self.X)[0]) if self.y[i] == 1])

       
        self.label0 = label0
        pl.plot(label0[:,0], label0[:,1],'ro')
        pl.plot(label1[:,0], label1[:,1],'go')

        pl.show()
