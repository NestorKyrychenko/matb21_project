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


def limit(func, a, b): #draws the graph of func around the point (a,b)
    ax = plt.figure().add_subplot(projection='3d')
    X, Y = np.meshgrid(np.linspace(a-0.5,a+0.5,100),np.linspace(b-0.5,b+0.5,100))
    Z = func(X,Y)

    # Plot the 3D surface
    ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                    alpha=0.3)

    # Plot projections of the contours for each dimension.  By choosing offsets
    # that match the appropriate axes limits, the projected contours will sit on
    # the 'walls' of the graph.
    ax.contour(X, Y, Z, zdir='z', offset=0, cmap='coolwarm')
    ax.contour(X, Y, Z, zdir='x', offset=a-0.5, cmap='coolwarm')
    ax.contour(X, Y, Z, zdir='y', offset=b-0.5, cmap='coolwarm')

    ax.set(xlim=(0.5, 1.5), ylim=(0.5, 1.5), zlim=(0, 1.25),
        xlabel='X', ylabel='Y', zlabel='Z')

    plt.show()

limit(f1,1,1)