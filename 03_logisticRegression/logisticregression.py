#!/usr/bin/python
###############################################################################
#
# logisticregression.py
#
# Implementation of Logistic Regression based Log-Likelihood maximation
#
# author: Michael Thomas, Jan Swoboda
# data: 2011-05-02
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

def discriminative(features, Y, beta):
    """ Compute discriminative function:
            
            f(y,x) = 0 if y==0, else features(x)^T * beta
    """
    
    # discriminative function
    #f = lambda y,x: y and x or 0

    # evaluate discriminative function for all y,features
    #dis_vector = mat([f(y.item(0),x.item(0)) for (y,x) in zip(Y,features*beta)]).T
    
    X = features * beta
    dis_vector = mat([x.item(0)*y.item(0) for (y,x) in zip(Y,X)]).T

    return dis_vector 


def classprob(features,Y,beta):
    """ Compute the class probablities 
            p(y=1|x) = sigma(f(y,x)) where sigma(z) = 1/(1+exp(-z))
    """

    sigma = lambda z: 1/(1+(exp(-z)))
    usigma = frompyfunc(sigma,1,1)

    prob = usigma(features*beta)

    return prob

def log_likelihood(prob,Y):
    
    lh = sum([y* log(p) + (1-y)*log(1-p) for (y,p) in zip(Y,prob) ])

    return lh

def newtonapprox(features,Y):

    # initial beta with zeros (m x 1)
    n,m = features.shape;
    beta = mat(zeros((m,1)))

    epsilon = 1.0e-9 

    maxiter = 2 # maximal number of iterations
   
    for i in range(maxiter):
        beta_new = newtonapprox_step(features,Y,beta)
        
        diff = abs(beta-beta_new).max()

        print "iteration: %d, diff: %.2f" % (i,diff)
        # if iteration converges, abbort
        if diff < epsilon:
            return beta_new

        beta = beta_new

    return beta
    
def newtonapprox_step(features,Y,beta_old):

    # compute new probabilities
    prob = classprob(features,Y,beta_old)

    # compute hessian weight matrix n x n with  x(p_i)*(1-p_i) on the diagonal
    weights = mat(diag([p.item(0)*(1-p.item(0)) for p in prob]))

    beta_new = beta_old + (features.T * weights *features).I * features.T * (Y-prob)

    return beta_new

class LogisticModel:

    beta = mat(0)

    def __init__(self,filename,getfeatures):
        self.train(filename,getfeatures)
   
    def train(self,filename,getfeatures):
        self.x,self.y = loaddata(filename)
   
        features = getfeatures(self.x)
   
        self.beta = newtonapprox(features,self.y)
    
    def plot(self,getfeatures):
        rows,cols = self.x.shape
        if cols == 2:
            self.plot2d(getfeatures)
            return
        if cols == 3:
            self.plot3d(getfeatures)
            return
        else:
            print "Only 1d and 2d input vectors supported for plotting"

    # plot predictions of testdata
    def plot2d(self,getfeatures):
   
        xmin,xmax = floor(self.x.min()),ceil(self.x.max())
        dataX = linspace(xmin,xmax,20)

        # seperate classes
        data_class0 = vstack([x for (x,y) in zip(self.x,self.y) if y==0])
        data_class1 = vstack([x for (x,y) in zip(self.x,self.y) if y==1])

        features = getfeatures(dataX)
   
#        predictions = features * self.beta
   
        #pl.plot(self.x,self.y,'ro')
        #pl.plot(dataX,predictions)
        pl.plot(data_class0[:,0], data_class0[:,1],'ro',color='red')
        pl.plot(data_class1[:,0], data_class1[:,1],'ro',color='green')
        

        # plot training data
        xs,ys = hsplit(self.x,[1])

        zs = self.y

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
        func = lambda x,y: getfeatures(mat([x,y])) * self.beta

        ufunction = frompyfunc(func,2,1)

        # compute z values
        Z = ufunction(X,Y)
    
        pl.contour(X,Y,Z,[0],color='blue')


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
        func = lambda x,y: getfeatures(mat([x,y])) * self.beta

        ufunction = frompyfunc(func,2,1)

        # compute z values
        Z = ufunction(X,Y)

        ax.plot_wireframe(X,Y,Z,alpha=0.4)
        
        fig.show()

        
        
