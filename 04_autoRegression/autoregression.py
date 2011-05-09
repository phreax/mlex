#!/usr/bin/python
###############################################################################
#
# linareg.py
#
# Implementation of Linear Regression based on Least Square Fit
#
# author: Mich_ael Thomas, Jan Swoboda
# data:  2011-05-09
#
###############################################################################

from numpy import *
import pylab as pl
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import cm
from linearregression import * 

def loadautodata(filename):

    data = loadtxt(filename)

    return data

# compute acceleration for i-th data vector
def compute_acceleration(data,offset,i):

    if i<offset: 
        vel0 = data[i,2]
        t0 = data[i,0]
    else:
        vel0 = data[i-offset,2]
        t0 = data[i-offset,0]

    if (i+offset) >= len(data):
        vel1 = data[i,2]
        t1 = data[i,0]
    else:
        vel1 = data[i+offset,2]
        t1 = data[i+offset,0]


    accel = (vel1-vel0)/(t1-t0)

    return accel

def getinput(data):
    """ compute input vector for auto regression """

    delay = 2

    X = []
    i = 0
    for line in data:
        X.append(delayed_data(data,delay,i))
        i += 1

    return mat(X)

def getoutput(data):
    """ compute output vector for auto regression """

    offset = 5
    y = []
    i=0
    for line in data:
        y.append(compute_acceleration(data,offset,i))
        i += 1

    return mat(y).T

def gettime(data):
    """ get time  vector from data """
    time = [xi[0] for xi in data]

    return mat(time).T


def delayed_data(data,delay,i):

    """ compute delayed data for i-th data vector (include data from the past """
   
    if i<delay:
        xi = [data[i,1], data[i,2], data[i,3], data[i,4], data[i,5]]
    else:
        xi = [data[i,1], data[i,2], data[i-delay,3], data[i-delay,4], data[i-delay,5]]

    return xi

class AutoRegression:

    def __init__(self,filename,getfeatures):

        data = loadautodata(filename)
        self.X = getinput(data)
        self.y = getoutput(data)
        self.time = gettime(data)

        self.model = LinearModel(getfeatures,self.X,self.y)

    def plot(self,getfeatures):

        predictions = self.model.predict(getfeatures,self.X)

        # plot output vecotor in time (acceleration)
        pl.plot(self.time,self.y,'r')
        pl.plot(self.time,predictions,'g')
        pl.show()






