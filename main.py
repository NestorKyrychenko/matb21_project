# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 08:53:13 2025

@author: Herman Plank
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin
from matplotlib import cm
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


def plot(func, a, b, delta = 0.5): #draws the graph of func around the point (a,b)
    ax = plt.figure().add_subplot(projection='3d')
    X, Y = np.meshgrid(np.linspace(a-delta,a+delta,100),np.linspace(b-delta,b+delta,100))
    Z = func(X,Y)

    # Plot the 3D surface
    ax.plot_surface(X, Y, Z, edgecolor='RoyalBlue', lw=0.5, rstride=8, cstride=8, alpha=0.3)
    z_min = np.nanmin(Z)
    ax.contourf(X,Y,Z,15, cmap='coolwarm',alpha=0.85, zdir='z', offset=z_min)
    ax.set(xlim=(a-delta, a+delta), ylim=(b-delta, b+delta), zlim=(z_min,np.nanmax(Z)),
        xlabel='X', ylabel='Y', zlabel='Z')
#plot(g1, 0, 0, delta=100)
#plot(g2, 0, 0, delta=3)

#%%Task 2

def sinfunc(x, y): #function sin(x + y)
    return np.sin(x + y)

def singrad(x, y): #analytical gradient of sin(x + y)
    return np.array([np.cos(x + y), np.cos(x + y)])

def grad(func, x, y, h = 10**(-5)): # returns apporx grad for function at (x,y) as an array
    return(np.array([(func(x+h,y)-func(x,y))/h,(func(x,y+h)-func(x,y))/h]))

def gradplot(x, y, h=10**(-6), d=2*np.pi): # plots the relative error between analytical and approx gradient of sin(x + y)
    pi = np.pi
    theta = np.arange(-2 * pi, 2 * pi+pi/2, step=(pi / 2))
    X, Y = np.meshgrid(np.linspace(x-d, x+d, 500), np.linspace(y-d, y+d, 500))
    Z1 = singrad(X, Y)
    Z2 = grad(sinfunc, X, Y, h)
    
    plt.subplot(121)
    plt.title(r'Relative error: $\frac{\delta x_{Approx} - \delta x_{Analytical}}{\delta x_{Analytical}}$' + f'  for $h = {h}$')
    plt.pcolormesh(X, Y, (Z2[0] - Z1[0])/Z1[0], cmap='binary')
    plt.colorbar(location='bottom')
    plt.xticks(theta, ['-2π', '-3π/2', 'π', '-π/2', '0', 'π/2', 'π', '3π/2', '2π'])
    plt.yticks(theta, ['-2π', '-3π/2', 'π', '-π/2', '0', 'π/2', 'π', '3π/2', '2π'])
    plt.subplot(122)
    plt.title(r'Relative error: $\frac{\delta y_{Approx} - \delta y_{Analytical}}{\delta y_{Analytical}}$' + f'  for $h = {h}$')
    plt.pcolormesh(X, Y, (Z2[1] - Z1[1])/Z1[1], cmap='binary')
    plt.colorbar(location='bottom')
    plt.xticks(theta, ['-2π', '-3π/2', 'π', '-π/2', '0', 'π/2', 'π', '3π/2', '2π'])
    plt.yticks(theta, ['-2π', '-3π/2', 'π', '-π/2', '0', 'π/2', 'π', '3π/2', '2π'])


def hessian(func, x, y, h = 10**(-6)): # returns a hessian matrix of func at x, y. for some reason doesnt always work w h = e-8
    f1 = lambda x, y: ((func(x+h,y)-func(x,y))/h)
    f2 = lambda x, y: ((func(x,y+h)-func(x,y))/h)
    return(np.array([[(f1(x+h,y)-f1(x,y))/h,(f1(x,y+h)-f1(x,y))/h],
                      [(f2(x+h,y)-f2(x,y))/h,(f2(x,y+h)-f2(x,y))/h]]))

def gradtest(x, y):
    h = np.linspace(-6,0, 500)
    errorx = (grad(sinfunc, x, y, 10**h)[0]  - singrad(x, y)[0])
    errory = (grad(sinfunc, x, y, 10**h)[1]  - singrad(x, y)[1])
    plt.subplot(121)
    plt.title(r'Approximated gradient minus analytical gradient of $f(x, y) = \sin(x + y)$ at $(x, y) = (\pi/4, \pi/4)$.')
    plt.plot(h, errorx)
    plt.ylabel(r'$\delta x_{Approx} - \delta x_{Analytical}$')
    plt.xlabel(r'$\log_{10}{h}$')
    plt.subplot(122)
    plt.title(r'Approximated gradient minus analytical gradient of $f(x, y) = \sin(x + y)$ at $(x, y) = (\pi/4, \pi/4)$.')
    plt.plot(h, errory)
    plt.ylabel(r'$\delta x_{Approx} - \delta x_{Analytical}$')
    plt.xlabel(r'$\log_{10}{h}$')

#gradtest(np.pi/4, np.pi/4)
