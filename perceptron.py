import numpy as np 
import random

# Activation function is taken as step function

# Percepton is classifier which classifes the input into two category (0/1)
# For generating multiple categories, a network of percepton should be used with multiple perceptron at output layer
# Following code trains a perceptron for a AND gate.
# Training a classifier for some set of input (in this example it is AND input) is similar to prediting a curve/line which will divide the whole hyperplane into two category (0/1) 
# We start with a random curve (by taking some random weights) and and then we check, are we classifying correctly each vector ?
# if not then we pick a random input vector and calculate the error for it.
# we correct our weights  ( which is called as learning ) using the following equation 
# here we multiply difference in error to vector because if error is positive then it means weight of all those vector parameter needs to be increased which are positive and other has to be decreased
 

class Perceptron:

	def __init__(self,size_of_vector):
		self.weights = np.random.rand(size_of_vector,1)
		self.bias = np.random.rand(1,1)

	def step(self,x):
		return np.piecewise(x,[x<0,x>=0],[0,1])

	def weightedSum(self,x):
		return self.bias+np.dot(x,self.weights)

	def train(self,X,Y,iterations):
		converges = False
		for i in range(iterations):
			P=self.predict(X)
			if not np.array_equal(P,Y):
				rint = random.randint(0,3) 
				x = X[rint]
				d = Y[rint]
				x=x[np.newaxis] #make it 2D to avoid dimensional errors 
				d=d[np.newaxis] #make it 2D to avoid dimensional errors
				y = self.predict(x)
				self.weights = self.weights + 0.5*(d[0][0]-y[0][0])*np.transpose(x) 	# eta = 0.5
				self.bias = self.bias + 0.5*(d[0][0]-y[0][0])
			else: 
				converges = True
				break
		if converges:
			print "Number of iterations to converge", i

	def predict(self,x):
		v = self.weightedSum(x)
		return self.step(v)

	def printWeight(self):
		print np.vstack((self.bias,self.weights) )
		


if __name__ =='__main__':
	print "A | B | A.B"
	X = np.array([[0,0],[0,1],[1,0],[1,1]])
	Y = np.array([[0],[0],[0],[1]])
	p = Perceptron(X.shape[1])
	print np.hstack( (X,Y) ) 
	p.train(X,Y,100)
	print "Trained weights:"
	p.printWeight()
	print "x1 | x2 | b+w1.x1+w2.x2"
	print np.hstack( (X,p.weightedSum(X)) )
	print "x1 | x2 | step(b+w1.x1+w2.x2)"	
	print np.hstack( (X,p.predict(X)) )
	print "it gets a classifier line which passes through"
	print -p.bias[0][0]/p.weights[0][0],"on x1's axis"
	print -p.bias[0][0]/p.weights[1][0],"on x2's axis"
	
	


