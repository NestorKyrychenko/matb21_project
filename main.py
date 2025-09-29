import matplotlib as plt
import numpy as np

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

def g1(x,y):
    return((x**2 + 3* y**2)*np.e**(-x**2-y**2))


