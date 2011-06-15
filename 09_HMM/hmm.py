# Implementation of the Baum-Welch Algorithm
# to estimate the parameter of a Hidden Markov Model
from numpy import *
import pylab as pl
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import cm
from itertools import izip

def gauss(x,mu,sigma):
    if type(mu) == type(sigma) == type(mat([])):
        p = sigma.shape[0]
        centered = x-mu
        res = 1/((2*pi)**(p/2.0) * linalg.det(sigma)**0.5) * exp(-0.5*centered.T*sigma.I*centered)
        return res[0,0] # return scalar
    
    if type(mu) == type(sigma) == type(float()):

        return 1/(sigma*(2*pi)**0.5) * exp(-0.5*((x-mu)/sigma)**2)

    # else
    raise TypeError("mu and covariance must either be matrices or float")

# recursive computation of the forward messages
def forward_msg(transition,forward,observed,t):

    # anchor
    if t==0:
       return mat(ones((transition.shape[0],1)))

    return transition*multiply(forward[t-1].T*observed[t-1])

# recursive computation of the backward messages
def backward_msg(transition,backward,observed,t):

    # anchor
    if t==backward.shape[0]-1:
       return mat(ones((transition.shape[0],1)))

    return transition.T*multiply(backward[t+1].T*observed[t+1])

def observed_msg(x,mu,sigma):

    # compute probability for each gaussian, convert
    # parameter into matrices,
    return mat([gauss(x,mat(param[0]).T,mat(param[1])) for param in izip(mu,sigma)]).T

class HMM:

    def load(self,datafile,hiddenfile=None):
        self.data = loadtxt(datafile)
        if(hiddenfile):
            self.hidden = loadtxt(hiddenfile)

    def __init__(self,nstates=2):
        self.nstates=nstates

        # correct parameter for testing
        self.transition = mat([[.99,.01],[.01,.99]])
        self.mu = [[0.0],[1.0]]
        self.sigma = [[1.0],[1.0]]

    # baum welch expecation step
    def e_step(self):


        observed = [observed_msg(x,self.mu,self.sigma) for x in self.data]

        # init messages
        forward = [ mat(zeros(observed[0].shape)) i in size(observed)]
        backward = [ mat(zeros(observed[0].shape)) i in size(observed)]

        for t in xrange(size(observed)):
            forward[t] = forward_msg(self.transition,forward,observed,t)
        
        for t in reversed(xrange(size(observed))):
            backward[t] = backward_msg(self.transition,backward,observed,t)
        

        # compute posteriors 
        self.posterior = [reduce(multiply,msg) for msg in izip(forward,observed,backward)]
        self.posterior_pair = zeros(self.data.shape[0],self.nstates)

        






      
