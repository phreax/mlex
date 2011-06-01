from numpy import *
import pylab
import sys

#
# get list of k-1 nearest neighbors 
#
# @param: weightet Adjacency Graph W
#         number of neighbors k
#
# @return: nearest neighbor list indizes for the data set
#          
def nearest_neighbors(W,k):
    nneighbors = []

    for i in range(W.shape[0]):
        nearest = []
        row = W[i].copy()
        # k nearest neighbors decided on maximum [0:1]
        for j in range(k):
            maxweight = max(row)
            # search for index of nearest neighbor point
            # based on weight in weight graph
            # since W is n x n - Matrix .
            # therefore it doesn't matter if we go thru 
            # cols or rows.
            for index in range(row.shape[0]):
                if maxweight == row[index]:
                    print maxweight
                    # no neighbors of one self?
                    if index == i:
                        nearest.append(index) # comment out this line if so
                        row[index] = 0
                    else:
                        # store index to nn's list of x_i
                        if maxweight == row[index]:
                            nearest.append(index)
                            row[index] = 0 # remove weight value
                                       # from row for next nearest
                                       # neighbor
        # don't forget to append the neighbors to nnearest neighbor matrix.
        nneighbors.append(nearest)

    kNN = zeros(W.shape)

    for i in range(nneighbors.__len__()):
        n = nneighbors[i]
        for j in n:
            kNN[i][j] = W[i][j]

    return kNN


#
# connectivity matrix
# set all nonzero fields to one
#
# @param: kNN Matrix
# @return: connectivity matrix
def connect(kNN):
    cmatrix = zeros(kNN.shape)
    for i in range(kNN.shape[0]):
        for j in range(kNN.shape[1]):
            if kNN[i][j] > 0:
                cmatrix[i][j] = 1
    return cmatrix
            
#
# compute diagonal matrix D
#
# @param: connectivity matrix
def diagonal_matrix(kNN):
    dmatrix = zeros(kNN.shape)
    for i in range(dmatrix.shape[0]):
        dmatrix[i][i] = sum(kNN[i])
    return dmatrix

#
# compute weights
#
# @param: data-matrix
# @return: weight-matrix
def weight(data):
    W = zeros((20,20))
    i = 0
    for x in data:
        j = 0
        for y in data:
           v = x-y
           norm = sqrt(v[0]**2 + v[1]**2)
           w = e ** -(norm/c)
           W[i][j] = w
           j = j +1
        i = i + 1
    return W

data = loadtxt('spiral_debug.txt')
c = 2
k = 5
W = weight(data)
kNN = nearest_neighbors(W,k)
cmatrix = connect(kNN)
dmatrix = diagonal_matrix(cmatrix)
L = dmatrix - kNN
U, s, V = linalg.svd(L)
eigenvalues, eigenvectors = linalg.eig(L)


#print kNN
#print kNN.shape
#print dmatrix
#print L
print "eigenvalues"
print eigenvalues
print "eigenvectors"
print eigenvectors

pylab.plot(data[:,0], data[:,0])
