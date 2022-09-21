###

#Testing Code for "An Unregularized Third-Order Newton Method"

#%%

#Packages

import numpy as np
import cvxpy as cvx
import numpy.linalg as LA
import matplotlib.pyplot as plt
import math


#%%

# Function here

def fx(X):

    x = X[0][0]

    y = X[1][0]

    return x**2 + 2*y**2 - 0.3*np.cos(3*pi*x) - 0.4*np.cos(4*pi*y)+0.7



def dx(X):

    x = X[0][0]

    y = X[1][0]

    return np.array([[2*x+0.9*pi*np.sin(3*pi*x)],\

                     [4*y+1.6*pi*np.sin(4*pi*y)]])



def d2x(X):

    x = X[0][0]

    y = X[1][0]

    ret = np.zeros((2,2))

    ret[0,0] = 2+2.7*pi**2*np.cos(3*pi*x)

    ret[0,1] = ret[1,0] = 0

    ret[1,1] = 4+6.4*pi**2*np.cos(4*pi*y)

    return ret



def d3x(X):

    x = X[0][0]

    y = X[1][0]

    ret = np.zeros((2,2,2))
    
    ret[0,0,0] = -8.1*pi**3*np.sin(3*pi*x)

    ret[0,0,1] = ret[0,1,0] = ret[1,0,0] = 0
    
    ret[1,1,0] = ret[1,0,1] = ret[0,1,1] = 0

    ret[1,1,1] = -25.6*pi**3*np.sin(4*pi*y)

    return ret;

#%%

# 3rd Order Newton Step

def min3(H,Q,b):
    n = len(b)

    constraints = []
    Y = cvx.Variable((n,n), symmetric=True)
    y = cvx.Variable((n,1))
    z = cvx.Variable((1,1))
    
    
    Hess = Q + sum([H[i,:,:]*y[i] for i in range(n)])
    L = cvx.Variable((n,1))
    
    constraints = constraints + [L[i] == cvx.trace(H[i,:,:]@Y) + Q[i,:]@y for i in range(n)]
    constraints = constraints + [Q[i,:]@y + cvx.trace(H[i,:,:]@Y)/2 + b[i] == 0 for i in range(n)]
    
    T = cvx.bmat([[Hess, L], [L.T, z]])
    YYT = cvx.bmat([[Y, y], [y.T, np.identity(1)]])
    constraints = constraints + [T >> 0]
    constraints = constraints + [YYT >> 0]
    
    prob = cvx.Problem(cvx.Minimize(cvx.trace(Q@Y)/2 + b.T@y + z/2),constraints)
    prob.solve(solver = cvx.SCS, verbose=False)
    
    px = prob.value
    return list([y.value,px,prob.status])



#%%

XMIN = -1;
XMAX = 1
YMIN = -1
YMAX = 1
RES2 = 501
RES3 = 101
MAX_ITS = 200
x_min = np.reshape(np.array([[0,.0]]),(2,1))

xlist2 = np.linspace(XMIN,XMAX,RES2)
ylist2 = np.linspace(YMIN,YMAX,RES2)
xlist3 = np.linspace(XMIN,XMAX,RES3)
ylist3 = np.linspace(YMIN,YMAX,RES3)


fractal_2 = np.zeros((RES2,RES2,3))
fractal_3 = np.zeros((RES3,RES3,3))

#%%

# Second Order

for xi in range(RES2):
    print(xi)
    for yi in range(RES2):
        x0 = np.array([xlist2[xi],ylist2[yi]])
        k = 0
        x_curr = np.reshape(x0,(2,1))
        while LA.norm(x_curr - x_min) > 10e-4 and k < MAX_ITS and LA.norm(dx(x_curr)) > .0001:
            x_curr = x_curr -LA.inv(d2x(x_curr)+.0001*np.identity(2))@dx(x_curr);
            k = k+1
        fractal_2[yi,xi,0:2] = np.reshape(x_curr,(1,2))
        fractal_2[yi,xi,2] = k
        
#%%

#The Fractal arrays

colors_2 = np.zeros((RES2,RES2))
for yi in range(RES2):
    for xi in range(RES2):
        if LA.norm(x_min.T-fractal_2[yi,xi,0:2]) < .001:
            colors_2[yi,xi] = 2
        else:
            colors_2[yi,xi] = 1
            
            
#%%

X, Y = np.meshgrid(xlist2, ylist2)
Z = X**2 + 2*Y**2 - 0.3*np.cos(3*np.pi*X) - 0.4*np.cos(4*np.pi*Y)+0.7


fig,ax=plt.subplots(1,1)
#cp = ax.contourf(X, Y, colors_2, cmap = plt.get_cmap('Greys'))
cp = ax.contourf(X, Y, colors_2)
ax.contour(X,Y,np.log(Z),colors='red')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


#%%

#Third Order

for xi in range(RES3):
    print(xi)
    for yi in range(RES3):
        x0 = np.array([xlist3[xi],ylist3[yi]])
        k = 0
        x_curr = np.reshape(x0,(2,1))
        while LA.norm(x_curr - x_min) > 10e-4 and k < MAX_ITS and LA.norm(dx(x_curr)) > .0001:
            out = min3(d3x(x_curr),d2x(x_curr),dx(x_curr))
            if 'infeasible' in out[2] or 'unbounded' in out[2]:
                break
            x_curr = x_curr + out[0];
            k = k+1
        fractal_3[yi,xi,0:2] = np.reshape(x_curr,(1,2))
        fractal_3[yi,xi,2] = k
    

#%%

colors_3 = np.zeros((RES3,RES3))            
for xi in range(RES3):
    for yi in range(RES3):
        if LA.norm(x_min.T-fractal_3[yi,xi,0:2]) < .001:
            colors_3[yi,xi] = 2
        else:
            colors_3[yi,xi] = 1
            
#%%

X, Y = np.meshgrid(xlist3, ylist3)
Z = X**2 + 2*Y**2 - 0.3*np.cos(3*np.pi*X) - 0.4*np.cos(4*np.pi*Y)+0.7


fig,ax=plt.subplots(1,1)
#cp = ax.contourf(X, Y, colors_3, cmap = plt.get_cmap('Greys'))
cp = ax.contourf(X, Y, colors_3)
ax.contour(X,Y,np.log(Z),colors='red')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()



