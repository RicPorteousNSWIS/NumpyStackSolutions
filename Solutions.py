# Importing relevant libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 



## Exercise 1
A = [[0.3,0.6,0.1],
[0.5,0.2,0.3],
[0.4,0.1,0.5]]

v = [[0.3333,0.3333,0.3333]]

A = np.array(A)
v = np.array(v)

print(A)
print(v)

dist = []
for i in xrange(25):
	v1= v.dot(A)
	dist.append( np.linalg.norm(v1-v) ) 
	v = v1
	print v

plt.plot((dist))
plt.show()



## Exercise 2
N = 1000
Y = 1000
tmp = []
for i in xrange(Y):
	tmp.append(np.random.random(N).sum())

plt.hist(tmp)
plt.show()

#Bonus: Theoretical means = N/2, theoretical variance = N/12



## Exercise 3:
# read in data
A = pd.read_csv('train.csv') 

# take the means for each label using group by 
B = A.groupby('label',as_index = False)[ A.columns[1:] ].mean()
B = B.as_matrix()
# plot each in a subplot

for i in xrange(10):
	im = B[i,1:].reshape(28,28)
	plt.subplot(3,4,i+1)
	plt.imshow(255-im,cmap = 'gray')

plt.show()

## Exercise 4
plt.subplot(1,2,1)
plt.imshow(255-B[0,1:].reshape(28,28),cmap = 'gray')

plt.subplot(1,2,2)
plt.imshow(255-np.rot90(B[0,1:].reshape(28,28)),cmap = 'gray')

plt.show()


## Exercise 5
def is_symmetric(a, tol = 1e-8):
	return np.allclose(a,a.T,atol = tol)

#test
nonsymmetric = np.array([[1 ,2],[3,4]])
symm = np.array([[1 ,2],[2,1]])

print "The non symmetric matrix output was %r" % is_symmetric(nonsymmetric)
print "The symmetric matrix output was %r " % is_symmetric(symm)

## Exercise 6
X = 2*np.random.random((1000,2))-0.5
W = np.round(abs(X)).astype(int)
color_mat = W[:,0]^W[:,1])
plt.scatter(X[:,0]-0.5,X[:,1]-0.5,c = color_mat)
plt.show()

## Exercise 7
N = 1000
R =np.array([10]*N + [5]*N)
theta= 2*np.pi*np.random.random(2*N)
Rr = R+(np.random.randn(len(R)))
X = Rr*np.cos(theta)
Y = Rr*np.sin(theta)
plt.scatter(X,Y,c = R)
plt.show()


# Exercise 8
N = 400
arms = 6
t = np.linspace(0,1,N) # parameter

Rr = []
theta = []
colormat = np.zeros((arms,N))
for i in xrange(arms):
	R_a = t
	theta_a= t*2*np.pi/arms+i*2*np.pi/arms
	Rr.append(R_a)
	theta.append(theta_a)
	if i % 2 ==0:
		colormat[i,:]=1

Rr = np.array(Rr)
theta = np.array(theta)

X = Rr*np.cos(theta)+np.random.randn(arms,N)*0.02
Y = Rr*np.sin(theta)+np.random.randn(arms,N)*0.02
plt.scatter(X,Y,c = colormat)
plt.show()




























