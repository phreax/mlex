#!/usr/bin/python
###############################################################################
#
# linareg.py
#
# Implementation of Linear Regression based on Least Square Fit
#
# author: Michael Thomas, Jan Swoboda
# data: 2011-04-24
#
###############################################################################

from numpy import *
import pylab as pl
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import cm

def loaddata(filename):

    # load data into an array
    data = loadtxt(filename)

    # get dimension of data array
    n,m = data.shape

    # split data into x and y matrices (first n columns are X, last column is y)
    X,y = hsplit(data,[m-1])

    return mat(X),mat(y)

def linearfeatures(X):

    # augment feature vector
    X = mat(map(lambda xi: append([1],xi),X))

    return X

def quadraticfeatures(X):

    # generate quadratic featues
    #X = concatenate([mat(append([[1],xi],(xi.T*xi).flatten())) for xi in X])
    X = concatenate([mat(append([xi],(xi.T*xi).flatten())) for xi in X])

    return linearfeatures(X)

def leastsquarefit(X,y):
    """ Compute optimal beta for input values X and output value Y """

    beta = (X.T*X).I * X.T * y

    return beta

class LinearModel:

    linearparam = mat(0)

    def __init__(self,filename,getfeatures):
        self.train(filename,getfeatures)
   
    def train(self,filename,getfeatures):
        self.x,self.y = loaddata(filename)
   
        X = getfeatures(self.x)
   
        self.linearparam = leastsquarefit(X,self.y)
    
    def plot(self,getfeatures):
        rows,cols = self.x.shape
        if cols == 1:
            self.plot2d(getfeatures)
            return
        if cols == 2:
            self.plot3d(getfeatures)
            return
        else:
            print "Only 1d and 2d input vectors supported for plotting"

    # plot predictions of testdata
    def plot2d(self,getfeatures):
   
        xmin,xmax = floor(self.x.min()),ceil(self.x.max())
        dataX = linspace(xmin,xmax,20)

        features = getfeatures(dataX)
   
        predictions = features * self.linearparam
   
        pl.plot(self.x,self.y,'ro')
        pl.plot(dataX,predictions)
        pl.show()
    
    def plot3d(self,getfeatures):

        fig = pl.figure()
        ax = p3.Axes3D(fig)

        # plot training data
        xs,ys = hsplit(self.x,[1])

        zs = self.y
        ax.scatter3D(xs,ys,zs,c='r')

        # get plotting range 
        mins = self.x.min(axis=0)
        maxs = self.x.max(axis=0)

        xmin,ymin = mins[0,0],mins[0,1]
        xmax,ymax = maxs[0,0],maxs[0,1]

        rangeX = linspace(xmin,xmax,20)
        rangeY = linspace(ymin,ymax,20)

        # compute grid matrices for X,Y
        X,Y = meshgrid(rangeX,rangeY)

        # generate universal function for computing z values
        func = lambda x,y: getfeatures(mat([x,y])) * self.linearparam

        ufunction = frompyfunc(func,2,1)

        # compute z values
        Z = ufunction(X,Y)

        ax.plot_wireframe(X,Y,Z,alpha=0.4)
        
        fig.show()

        
        
