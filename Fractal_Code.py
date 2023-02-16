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

def alpha_approx(Dx, D2x, D3x):
    
    n_dx = LA.norm(Dx)
    n = len(Dx)
    ns_d3x = [LA.norm(D3x[i,:,:]) for i in range(n)]
    n_d3x = LA.norm(ns_d3x,2)
    
    e_val, e_vec = LA.eigh(D2x)
    idx = np.argsort(e_val)
    e_val = e_val[idx]
    e_vec = e_vec[:,idx]
    e_min = np.min(e_val[0],0)
    
    return np.sqrt(3*n_dx * n_d3x) - e_min

#%%

# Second Order

RES2 = 501
MAX_ITS = 200
xlist2 = np.linspace(XMIN,XMAX,RES2)
ylist2 = np.linspace(YMIN,YMAX,RES2)
fractal_2 = np.zeros((RES2,RES2,3))

for xi in range(RES2):
    print(xi)
    for yi in range(RES2):
        x0 = np.array([xlist2[xi],ylist2[yi]])
        k = 0
        x_curr = np.reshape(x0,(2,1))
        while LA.norm(x_curr - x_min) > 10e-4 and k < MAX_ITS and LA.norm(dx(x_curr)) > .001:
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
Z = np.sin(X + Y) + (X - Y)**2 - 1.5*X + 2.5*Y + 3


fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, colors_2, cmap = plt.get_cmap('Greys'))
ax.contour(X,Y,np.log(Z),colors='red')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


#%%

#Third Order without Levenberg Marquardt

RES3 = 101
MAX_ITS = 200
xlist3 = np.linspace(XMIN,XMAX,RES3)
ylist3 = np.linspace(YMIN,YMAX,RES3)
fractal_3 = np.zeros((RES3,RES3,3))

for xi in range(RES3):
    print(xi)
    for yi in range(RES3):
        x0 = np.array([xlist3[xi],ylist3[yi]])
        k = 0
        x_curr = np.reshape(x0,(2,1))
        while LA.norm(x_curr - x_min) > 10e-4 and k < MAX_ITS and LA.norm(dx(x_curr)) > .001:
            out = min3(d3x(x_curr),d2x(x_curr),dx(x_curr))
            if 'infeasible' in out[2] or 'unbounded' in out[2]:
                fractal_3[yi,xi,2] = -1
                break
            x_curr = x_curr + out[0];
            k = k+1
        fractal_3[yi,xi,0:2] = np.reshape(x_curr,(1,2))
        fractal_3[yi,xi,2] = k

#%%  

#Third Order with Levenberg Marquardt

RES3 = 101
MAX_ITS = 200
xlist3 = np.linspace(XMIN,XMAX,RES3)
ylist3 = np.linspace(YMIN,YMAX,RES3)
fractal_3 = np.zeros((RES3,RES3,3))

for xi in range(RES3):
    print(xi)
    for yi in range(RES3):
        x0 = np.array([xlist3[xi],ylist3[yi]])
        k = 0
        x_curr = np.reshape(x0,(2,1))
        while k < MAX_ITS and LA.norm(dx(x_curr)) > .001 and LA.norm(x_curr) < 25:
            D3x = d3x(x_curr)
            D2x = d2x(x_curr)
            Dx = dx(x_curr)
            out = min3(D3x,D2x, Dx)
            if 'infeasible' in out[2] or 'unbounded' in out[2]:
                out = min3(D3x,D2x + alpha_approx(Dx, D2x, D3x)*np.eye(2), Dx)
            x_curr = x_curr + out[0];
            k = k+1
        fractal_3[yi,xi,0:2] = np.reshape(x_curr,(1,2))
        fractal_3[yi,xi,2] = k

#%%

colors_3 = np.zeros((RES3,RES3))      
num_mins = x_min.shape[0]
for xi in range(RES3):
    for yi in range(RES3):
        for i in range(num_mins):
            if LA.norm(x_min[i,:]-fractal_3[yi,xi,0:2]) < .001:
                colors_3[yi,xi] = num_mins - i
                break
        colors_2[yi,xi] = 0
            
#%%

X, Y = np.meshgrid(xlist3, ylist3)
Z = fX(X,Y)


fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, colors_3, cmap = plt.get_cmap('Greys'))
ax.contour(X,Y,np.log(Z),colors='red')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()



