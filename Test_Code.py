###

#Testing Code for "An Unregularized Third-Order Newton Method"

#%%

#Packages

import numpy as np
import cvxpy as cvx
import numpy.linalg as LA
import matplotlib.pyplot as plt

#%%

# Function here

def fx(X):
    x = X[0][0]
    y = X[1][0]
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

def dx(X):
    x = X[0][0]
    y = X[1][0]
    return np.array([[2*x*(y**6 + y**4 - 2*y**3 - y**2 - 2*y + 3) + 5.25*y**3 + 4.5*y**2 + 3*y - 12.75],\
                     [6*x*(x*(y**5 + (2/3)*y**3 - y**2 - (1/3)*y - 1/3) + 2.625*y**2 + 1.5*y + .5)]])

def d2x(X):
    x = X[0][0]
    y = X[1][0]
    ret = np.zeros((2,2))
    ret[0,0] = 2*y**6 + 2*y**4 - 4*y**3 - 2*y**2 - 4*y + 6
    ret[0,1] = ret[1,0] = 12*x*y**5 + 8*x*y**3 - 12*x*y**2 - 4*x*y - 4*x + 15.75*y**2 + 9*y + 3
    ret[1,1] = 30*x**2*y**4 + 12*x**2*y**2 - 12*x**2*y - 2*x**2 + 31.5*x*y + 9*x
    return ret

def d3x(X):
    x = X[0][0]
    y = X[1][0]
    ret = np.zeros((2,2,2))
    ret[0,0,1] = ret[0,1,0] = ret[1,0,0] = 12*y**5 + 8*y**3 - 12*y**2 - 4*y - 4
    ret[1,1,0] = ret[1,0,1] = ret[0,1,1] = 60*x*y**4 + 24*x*y**2 - 24*x*y - 4*x + 31.5*y + 9
    ret[1,1,1] = 120*x**2*y**3 + 24*x**2*y - 12*x**2 + 31.5*x
    return ret;



#%%

# 3rd Order Newton

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
    prob.solve(solver = cvx.SCS, verbose=False)#, mosek_params = MPs)
    
    return list([y.value,prob.value,prob.status])

def N3O(x0, max_its):
    n = len(x0)
    its = np.zeros((max_its,2))
    d_gap = np.zeros((max_its,1))
    its[0,:] = x0
    
    for i in range(1,max_its):
        x_curr = np.reshape(its[i-1,:], (n,1));
        if LA.norm(dx(x_curr)) < 10e-6:
            break
        out = min3(d3x(x_curr),d2x(x_curr),dx(x_curr))
        if 'infeasible' in out[2] or 'unbounded' in out[2]:
            break
        x_next = x_curr + out[0];
        its[i,:] = x_next.T;
        d_gap[i] = out[1];
    its = its[0:i-1,:]

    return its, i-1, d_gap

#%%

# 2nd Order Newton

def N2O(x0, max_its):
    n = len(x0)
    its = np.zeros((max_its,2))
    its[0,:] = x0
    x_curr = x0
    
    for i in range(1,max_its):
        x_curr = np.reshape(its[i-1,:], (n,1));
        if LA.norm(dx(x_curr)) < 10e-4:
            break
        x_next = x_curr -LA.inv(d2x(x_curr))@dx(x_curr);
        its[i,:] = x_next.T;
    its = its[0:i-1,:]
    
    return its, i-1


#%%

# Damped Newton's Method

def NDamp(x0, max_its):
    beta = .9
    alpha = .5
    
    n = len(x0)
    its = np.zeros((max_its,2))
    its[0,:] = x0
    x_curr = x0
    f_curr = fx(x_curr)
    dx_curr = dx(x_curr)
    
    for i in range(1,max_its):
        x_curr = np.reshape(its[i-1,:], (n,1));
        if LA.norm(dx(x_curr)) < 10e-4:
            break
        dirx = LA.inv(d2x(x_curr))@dx(x_curr);
        while fx(x_curr - dirx) >= f_curr + alpha*dx_curr.T@dirx:
            dirx = beta*dirx
        
        x_next = x_curr - dirx;
        its[i,:] = x_next.T;
    its = its[0:i-1,:]
    
    return its, i-1
    

#%%

# Gradient Descent

def gradD(x0, max_its):
    n = len(x0)
    its = np.zeros((max_its,2))
    its[0,:] = x0
    x_curr = x0
    
    for i in range(1,max_its):
        x_curr = np.reshape(its[i-1,:], (n,1));
        if LA.norm(dx(x_curr)) < 10e-4:
            break
        x_next = x_curr - dx(x_curr);
        its[i,:] = x_next.T;
    its = its[0:i-1,:]
    
    return its, i-1
    
    
#%%

# Gradient Descent with exact quadratic line search
    
def gradELS(x0, max_its):
    n = len(x0)
    its = np.zeros((max_its,2))
    its[0,:] = x0
    
    for i in range(1,max_its):
        x_curr = np.reshape(its[i-1,:], (n,1));
        if LA.norm(dx(x_curr)) < 10e-4:
            break
        
        Q = d2x(x_curr)
        b = dx(x_curr)
        SS = b.T@b/(b.T@Q@b)
        
        x_next = x_curr - SS*dx(x_curr);
        its[i,:] = x_next.T;
    its = its[0:i-1,:]
    
    return its, i-1


#%%

#Running

x_init = np.array([2,0])
max_its = 5000

x_3ON, its_3ON, d_gaps = N3O(x_init, max_its)
x_2ON, its_2ON = N2O(x_init, max_its)
x_Damp, its_Damp = NDamp(x_init, max_its)
x_GD, its_GD = gradD(x_init, max_its)
x_ELS, its_ELS = gradELS(x_init, max_its)


#%%

#Plotting

XMIN = 1.5
XMAX = 3.5
YMIN = -.2
YMAX = .6
RES = 101

xlist = np.linspace(XMIN, XMAX, RES)
ylist = np.linspace(YMIN, YMAX, RES)
X, Y = np.meshgrid(xlist, ylist)
FN = np.log((1.5-X+X*Y)**2 + (2.25-X+X*Y**2)**2 + (2.625-X+X*Y**3)**2)

its = x_2ON


fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, FN)
ax.plot(its[:,0],its[:,1], color = 'red')
#fig.colorbar(cp) # Add a colorbar to a plot
#ax.set_title('Filled Contours Plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

















