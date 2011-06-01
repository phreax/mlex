# create a table of
# the k nearest neighbor of
# a the given data points X

from numpy import *
from operator import itemgetter

def squaredDist(x,y):
    return sum(map(square, (x-y)))


class KNN:

    def __init__(self,X,k):
        self.X = X;
        self.k = k;

        self.knnTable = {}

        self.createTable()

    def createTable(self):

        """Compute k nearest neigbors and save
        them in a dictonary together with the distances (index as key)
        """

        # very inefficient O(n^3)
        i=0
        length = shape(self.X)[0]
        for x in self.X:
            # compute distances to all vectors if i!=j
            distances = [(squaredDist(x,self.X[j]),j) for j in range(length) if i != j ]
           
            # sort by distance and pick first k vectors
            knearest = sorted(distances,key=itemgetter(0))[:self.k]
            # pick k smallest
            self.knnTable[i] = knearest
            i +=1
    
    def getNN(self, i):
        """return indeces of the k nearest neighors of point i"""
        return [n[1] for n in self.knnTable[i]]

        
    def getNNDist(self,i):
        """ get distances to the k nearest neighbors """

        return [n[0] for n in self.knnTable[i]]
            



