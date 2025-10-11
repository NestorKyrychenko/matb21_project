import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d

#Task 1

# defining the functions we want to find the limits of
def f1(x,y): # want to evaluate limit at (1,1)
    return((x**2-x*y)/(x**2-y**2))

def f2(x,y): # want to evaluate limit at (0,0)
    return((x**2+y**2)/(x**2+x*y+y**2))

def f3(x,y): # want to evaluate limit at (0,-1)
    return((np.sin(x+x*y)-x-x*y)/(x*(y+1))**3)

# defining the functions we want to find the extrema of

def g1(x,y):
    return(8*x*y - 4*x**2*y - 2*x*y**2 + x**2*y**2)

def g2(x,y):
    return((x**2 + 3* y**2)*np.e**(-x**2-y**2))


def limit(func, a, b, delta = 0.5): #draws the graph of func around the point (a,b)
    ax = plt.figure().add_subplot(projection='3d')
    X, Y = np.meshgrid(np.linspace(a-delta,a+delta,100),np.linspace(b-delta,b+delta,100))
    Z = func(X,Y)

    # Plot the 3D surface
    ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                    alpha=0.3)

    z_min = np.nanmin(Z)
    ax.contourf(X,Y,Z,15, cmap='coolwarm',alpha=0.85, zdir='z', offset=z_min)
    ax.set(xlim=(a-delta, a+delta), ylim=(b-delta, b+delta), zlim=(z_min,np.nanmax(Z)),
        xlabel='X', ylabel='Y', zlabel='Z')
    plt.show()

def grad(func, x, y, h = 10**(-6)): # returns apporx grad for function at (x,y) as an array
    return(np.array([[(func(x+h,y)-func(x,y))/h,(func(x,y+h)-func(x,y))/h]]))

def hessian(func, x, y, h = 10**(-6)): # returns a hessian matrix of func at x, y. for some reason doesnt always work w h = e-8
    f1 = lambda x, y: ((func(x+h,y)-func(x,y))/h)
    f2 = lambda x, y: ((func(x,y+h)-func(x,y))/h)
    return(np.array([[(f1(x+h,y)-f1(x,y))/h,(f1(x,y+h)-f1(x,y))/h],
                      [(f2(x+h,y)-f2(x,y))/h,(f2(x,y+h)-f2(x,y))/h]]))

limit(f3,0,-1,delta=0.1)
