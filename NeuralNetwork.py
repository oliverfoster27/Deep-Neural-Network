import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from scipy import optimize
from numpy import array

#set print options for the analyzer
float_formatter = lambda x: "%.4f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

X=[]
Y=[]

firstline=True
for line in open('CoolPlot2.csv'):
    if firstline:
        firstline=False
        l1,l2,l3=line.split(',')
    else:
        x1,x2,y=line.split(',')
        X.append([float(x1),float(x2)])
        Y.append(float(y))

X=np.array(X)
Y=np.array(Y)
#transpose the 1D array (transpose function won't work for 1D)
Y=Y[np.newaxis,:].T

#X=np.array(([3,5],[5,1],[10,2]))
#Y=np.array(([75],[82],[93]))

#Scale our data
max_X=np.amax(X,axis=0)
max_Y=np.amax(Y,axis=0)

Y_scale=0
if np.amin(Y,axis=0)[0]<0:
    Y_scale=np.amin(Y,axis=0)[0]

X=X/max_X
Y=(Y-Y_scale)/(max_Y-Y_scale)

# Whole Class with additions:
class Neural_Network(object):
    def __init__(self):        

        #Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1

        self.hiddenLayerSizes=np.array([3])
        self.numberHiddenLayers=len(self.hiddenLayerSizes)
        
        #Weights (parameters)
        self.W=[
            array(np.random.randn(self.inputLayerSize,self.hiddenLayerSizes[0]))
        ]
        for counter in range(1,self.numberHiddenLayers):
            self.W.append(array(np.random.randn(self.hiddenLayerSizes[counter-1],self.hiddenLayerSizes[counter])))
        self.W.append(array(np.random.randn(self.hiddenLayerSizes[self.numberHiddenLayers-1],self.outputLayerSize)))
        
    def forward(self, X):
        #Propogate inputs though network

        self.z=[
            array(np.dot(X,self.W[0]))
        ]
        self.a=[
            array(self.sigmoid(self.z[0]))
        ]
        for counter in range(1,self.numberHiddenLayers):
            self.z.append(array(np.dot(self.a[counter-1],self.W[counter])))
            self.a.append(self.sigmoid(array(self.z[len(self.z)-1])))
        self.z.append(array(np.dot(self.a[len(self.a)-1],self.W[len(self.W)-1])))
        yHat=self.sigmoid(self.z[len(self.z)-1])

        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        np.seterr(all='print')
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        np.seterr(all='print')
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)

        delta=[
            array(np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z[len(self.z)-1])))
        ]
        dJdW=[
            array(np.dot(array(self.a[len(self.a)-1]).T,array(delta[0])))
        ]
        for counter in range(1,self.numberHiddenLayers):
            delta.append(array(np.dot(array(delta[len(delta)-1]),array(self.W[len(self.W)-counter]).T)*self.sigmoidPrime(self.z[len(self.z)-1-counter])))
            dJdW.append(array(np.dot(array(self.a[len(self.a)-1-counter]).T,array(delta[len(delta)-1]))))
            
        delta.append(array(np.dot(array(delta[len(delta)-1]),array(self.W[1]).T)*self.sigmoidPrime(self.z[0])))
        dJdW.append(array(np.dot(X.T, array(delta[len(delta)-1]))))
        
        return dJdW

    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:

        params=np.concatenate((array(self.W[0]).ravel(),array(self.W[1]).ravel()))
        for counter in range(2,len(self.W)):
            params=np.concatenate((params,array(self.W[counter]).ravel()))

        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.

        W1_start=0
        W_end=[
            self.hiddenLayerSizes[0]*self.inputLayerSize
        ]
        self.W[0]=array(np.reshape(params[W1_start:W_end[0]],(self.inputLayerSize,self.hiddenLayerSizes[0])))
        for counter in range(1, self.numberHiddenLayers):
            W_end.append(W_end[len(W_end)-1]+self.hiddenLayerSizes[counter-1]*self.hiddenLayerSizes[counter])
            self.W[counter]=array(np.reshape(params[W_end[counter-1]:W_end[counter]], (self.hiddenLayerSizes[counter-1],self.hiddenLayerSizes[counter])))
        W_end.append(W_end[len(W_end)-1]+self.hiddenLayerSizes[len(self.hiddenLayerSizes)-1]*self.outputLayerSize)
        self.W[len(self.W)-1]=array(np.reshape(params[W_end[len(W_end)-2]:W_end[len(W_end)-1]], (self.hiddenLayerSizes[len(self.hiddenLayerSizes)-1],self.outputLayerSize)))
        
        #W1_start = 0
        #W1_end = self.hiddenLayerSize * self.inputLayerSize
        #self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        #W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        #self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW = self.costFunctionPrime(X, y)
        grads=array(np.concatenate((array(dJdW[len(dJdW)-1]).ravel(),array(dJdW[len(dJdW)-2]).ravel())))
        for counter in range(len(dJdW)-3,-1,-1):
            grads=array(np.concatenate((array(grads),array(dJdW[counter].ravel()))))

        return grads


class trainer(object):

    #this class trains the neural network by setting the weights
    
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))   
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        
        return cost, grad
        
    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 20000, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res

class analyze(object):

    #this class can be called to analyze the data once the network has been trained
    
    def __init__(self,N):
        self.N=N

    def scaledInput(self,X):
        return np.concatenate((max_X[:1]*X[:,:1],max_X[1:]*X[:,1:]),axis=1)

    def scaledOutput(self,Y):
        return max_Y*Y

    def outputError(self,X,Y):
        return abs((Y-self.N.forward(X))/Y)*100

    def Plot3D(self):

        #Test network for various combinations of sleep/study:
        firstvar = np.linspace(0, max_X[:1],1000)
        secondvar = np.linspace(0, max_X[1:],1000)

        #Normalize data (same way training data way normalized)
        firstvarnorm = firstvar/max_X[:1]
        secondvarnorm = secondvar/max_X[1:]

        #Create 2-d versions of input for plotting
        a, b  = np.meshgrid(firstvarnorm, secondvarnorm)

        #Join into a single input matrix:
        allInputs = np.zeros((a.size, 2))
        allInputs[:, 0] = a.ravel()
        allInputs[:, 1] = b.ravel()

        allOutputs = N.forward(allInputs)

        yy = np.dot(secondvar.reshape(1000,1), np.ones((1,1000)))
        xx = np.dot(firstvar.reshape(1000,1), np.ones((1,1000))).T

        #3D plot:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        surf = ax.plot_surface(xx, yy, ((max_Y[0]-Y_scale)*allOutputs.reshape(1000, 1000))+Y_scale, \
                               cmap=cm.jet)

        ax.set_xlabel(l1)
        ax.set_ylabel(l2)
        ax.set_zlabel(l3)

        plt.show()

    def Query(self,x1,x2):
        firstvar=x1/max_X[:1]
        secondvar=x2/max_X[1:]
        XQ=np.array(([firstvar[0],secondvar[0]]))
        return N.forward(XQ)*(max_Y[0]-Y_scale)+Y_scale

#Now call all the classes to perform the operations after running
N=Neural_Network()
T=trainer(N)
T.train(X,Y)
A=analyze(N)
A.Plot3D()
