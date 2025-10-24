#!/usr/bin/env python3
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
import scipy as scipy
import time


# Task 1

# defining the functions we want to find the limits of
def f1(x, y):  # want to evaluate limit at (1,1)
    return ((x**2-x*y)/(x**2-y**2))


def f2(x, y):  # want to evaluate limit at (0,0)
    return ((x**2+y**2)/(x**2+x*y+y**2))


def f3(x, y):  # want to evaluate limit at (0,-1)
    return ((np.sin(x+x*y)-x-x*y)/(x*(y+1))**3)

# defining the functions we want to find the extrema of


def g1(x, y):
    return (8*x*y - 4*x**2*y - 2*x*y**2 + x**2*y**2)


def g2(x, y):
    return ((x**2 + 3 * y**2)*np.e**(-x**2-y**2))


def plot(func, a, b, delta=0.5, point=False):  # draws the graph of func around the point (a,b)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    X, Y = np.meshgrid(np.linspace(a-delta, a+delta, 200),
                       np.linspace(b-delta, b+delta, 200))
    Z = func(X, Y)
    # Plot the 3D surface
    ax.plot_surface(X, Y, Z, edgecolor='black', cmap='managua', lw=0.5,
                    rstride=8, cstride=8, alpha=0.3)
    z_min = np.nanmin(Z)
    ax.contourf(X, Y, Z, 100, cmap='managua', alpha=0.85, zdir='z', offset=z_min)
    ax = fig.add_subplot(1, 2, 2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    contour = ax.contourf(X, Y, Z, 100, cmap='managua', alpha=0.85, zdir='z', offset=z_min)
    fig.colorbar(contour, location='bottom')
    if point:
        ax.scatter(a, b, c='w', marker='*')
    
#eq 1:
plot(f1, 1, 1, delta=1, point=True)
plot(f2, 0, 0, delta=3, point=True)
plot(f3, 0, -1, delta=3, point=True)
#eq 2:
plot(g1, 0, 0, delta=100)
plot(g2, 0, 0, delta=3)

# %%Task 2


def sinfunc(x, y):  # function sin(x + y)
    return np.sin(x + y)


def singrad(x, y):  # analytical gradient of sin(x + y)
    return np.array([np.cos(x + y), np.cos(x + y)])


def grad(func, x, y, h=10**(-6)):  # returns approx grad for function at (x,y) as an array
    return (np.array([(func(x+h, y)-func(x, y))/h, (func(x, y+h)-func(x, y))/h]))

# plots the relative error between analytical and approx gradient of sin(x + y) around (x,y)
def gradplot(x, y, h=10**(-6), d=2*np.pi):
    fig, ax = plt.subplots(1,2)
    pi = np.pi
    theta = np.arange(-2 * pi, 2 * pi+pi/2, step=(pi / 2))
    X, Y = np.meshgrid(np.linspace(x-d, x+d, 500), np.linspace(y-d, y+d, 500))
    Z1 = singrad(X, Y)
    Z2 = grad(sinfunc, X, Y, h)
    plt.subplot(121)
    ax[0].set_title(r'Relative error: $|\frac{\delta x_{Approx} - \delta x_{Analytical}}{\delta x_{Analytical}}|$' + f'  for $h = {h}$')
    ax[0].set_xticks(theta, ['-2π', '-3π/2', 'π', '-π/2',
               '0', 'π/2', 'π', '3π/2', '2π'])
    ax[0].set_yticks(theta, ['-2π', '-3π/2', 'π', '-π/2',
               '0', 'π/2', 'π', '3π/2', '2π'])
    plot = ax[0].pcolormesh(X, Y, np.abs((Z2[0] - Z1[0])/Z1[0]), cmap='binary')
    fig.colorbar(plot, location='bottom')
    
    ax[1].set_title(r'Relative error: $|\frac{\delta y_{Approx} - \delta y_{Analytical}}{\delta y_{Analytical}}|$' + f'  for $h = {h}$')
    ax[1].set_xticks(theta, ['-2π', '-3π/2', 'π', '-π/2',
               '0', 'π/2', 'π', '3π/2', '2π'])
    ax[1].set_yticks(theta, ['-2π', '-3π/2', 'π', '-π/2',
               '0', 'π/2', 'π', '3π/2', '2π'])
    plot = ax[1].pcolormesh(X, Y, np.abs((Z2[1] - Z1[1])/Z1[1]), cmap='binary')
    fig.colorbar(plot, location='bottom')
    
def gradtest(x, y): #plots the absolute error between approximated gradient and actual at a point (x,y) for different h
    fig, ax = plt.subplots(1,2)
    h = np.linspace(-6, 0, 500)
    errorx = abs(grad(sinfunc, x, y, 10**h)[0] - singrad(x, y)[0])
    errory = abs(grad(sinfunc, x, y, 10**h)[1] - singrad(x, y)[1])
    
    ax[0].set_title(r'Approximated gradient minus analytical gradient of $f(x, y) = \sin(x + y)$ at $(x, y) = (\pi/4, \pi/4)$.')
    ax[0].set_xlabel(r'$\log_{10}{h}$')
    ax[0].set_ylabel(r'$|\delta x_{Approx} - \delta x_{Analytical}|$')
    ax[0].plot(h, errorx, color='black')
    
    ax[1].set_title(r'Approximated gradient minus analytical gradient of $f(x, y) = \sin(x + y)$ at $(x, y) = (\pi/4, \pi/4)$.')
    ax[1].set_xlabel(r'$\log_{10}{h}$')
    ax[1].set_ylabel(r'$|\delta y_{Approx} - \delta y_{Analytical}|$')
    ax[1].plot(h, errory, color='black')

# returns a hessian matrix of func at x, y. for some reason doesnt always work w h = e-8
def hessian(func, x, y, h=10**(-6)):
    def f1(x, y): return ((func(x+h, y)-func(x, y))/h)
    def f2(x, y): return ((func(x, y+h)-func(x, y))/h)
    return (np.array([[(f1(x+h, y)-f1(x, y))/h, (f1(x, y+h)-f1(x, y))/h],
                     [(f2(x+h, y)-f2(x, y))/h, (f2(x, y+h)-f2(x, y))/h]]))

def sinhessian(x,y):
    return np.array([[-np.sin(x+y), -np.sin(x+y)], [-np.sin(x+y), -np.sin(x+y)]])


gradplot(0, 0, h=10**-6)
gradtest(np.pi/4, np.pi/4)

p = 100
x = np.linspace(-2*np.pi, 2*np.pi, p)
y = np.linspace(-2*np.pi, 2*np.pi, p)
X, Y = np.meshgrid(x, y)
Z = np.zeros([p, p])
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i,j] = np.abs(np.linalg.det(hessian(sinfunc, i, j)) - np.linalg.det(sinhessian(i, j)))

fig, ax = plt.subplots()
hplot = ax.pcolormesh(X, Y, Z, cmap='binary')
ax.set_title('Absolute values of the difference of the determinant of the approximated Hessian and the analytical Hessian of sin(x+y)')
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.colorbar(hplot, location='right')

#using g(x,y) = (x**2 + 3 * y**2)*np.e**(-x**2-y**2) at min point (0,0):
    
print('The gradient of g(x,y) at (0,0) is: ', grad(g2, 0, 0), '\n and the Hessian matrix of g at (0,0) is: \n', hessian(g2, 0, 0))
    

#task 3

def F(x,y,z):
    return(x+2*y+z+np.e**(2*z)-1)


x = np.linspace(-1,1,100)
y = np.linspace(-1,1,100)
X, Y = np.meshgrid(x,y)
Z = np.zeros_like(X)


for i in range(X.shape[0]):
    for j in range(X.shape[1]): # iterate through the grid and do fszolve for z for each point
        Z[i,j] = scipy.optimize.fsolve(lambda z: F(X[i,j],Y[i,j],z),0)[0]

ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8, alpha=0.3) # plot the found surfacer

f = lambda a,b: scipy.interpolate.RegularGridInterpolator((x, y), Z)([[a,b]])[0] # interpolate so that we have z(x,y) as a callable function

g = grad(f,0,0)
h = hessian(f,0,0) # we will use these every time in coefficients for P2, so calculating them here saves time
def P2(x,y):
    point = [[x,y]]
    return((f(0,0) + (point@g)[0] + ((point)@(np.cross(h,point)))[0]*0.5)[0])

Z1 = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]): # we find the Z values for the P2 approximation
        Z1[i,j] = P2(X[i,j],Y[i,j])

ax.plot_surface(X,Y, Z1, edgecolor='red', lw=0.5, rstride=8, cstride=8, alpha=0.3)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Graphs of the implicit function (blue) and its Taylor approximation (red)')

plt.figure(figsize=(6,5))
pcm = plt.pcolormesh(X,Y,abs(Z-Z1), cmap = 'inferno') # heat map of the error in approximation
plt.colorbar(pcm, label='|Error|')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Absolute Error between f(x,y) and Taylor Approximation')

plt.show()

#task 4

def task4_f(x,y):                                             # Defining Himmelblau function
    return((x**2+y-11)**2+(x+y**2-7)**2)

def gradient_descent(f, x, y, alpha = 0.01, steps = 20):         # Implementing gradient descent algorithm
    X = np.transpose([[x,y]])
    for i in range (steps):
        X_new = X - alpha*grad(f,X[0,0],X[1,0])
        X = X_new
    return(X_new)

x = np.linspace(-5,5,100)
y = np.linspace(-5,5,100)
X, Y = np.meshgrid(x,y)
Z = task4_f(X,Y)                 # Setting up the space in which we plot


ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,alpha=0.3)
ax.contourf(X,Y,Z,30, cmap='coolwarm',alpha=0.85, zdir='z', offset=np.nanmin(Z))
ax.set(xlim=(-5, 5), ylim=(-5, 5), zlim=(np.nanmin(Z),np.nanmax(Z)),xlabel='X', ylabel='Y', zlabel='Z') # Add surface plot and contour to figure

for i in range(1,20):
    x, y = gradient_descent(task4_f,0,0, steps = i)
    # applying algorithm and marking progression, starting at (0,0)
    ax.scatter(x,y,task4_f(x,y), s = 20, color = 'red',depthshade=False)

for i in range(1,20):
    x, y = gradient_descent(task4_f,0.2,-4, steps = i)
    # applying algorithm and marking progression, starting at (0.2,-4)
    ax.scatter(x,y,task4_f(x,y), s = 20, color = 'green',depthshade=False)

plt.show()

grad4_1 = grad(task4_f, gradient_descent(task4_f, 0,0)[0], gradient_descent(task4_f, 0,0)[1])
Hessian4_1 = hessian(task4_f, gradient_descent(task4_f, 0,0)[0], gradient_descent(task4_f, 0,0)[1])
grad4_2 = grad(task4_f, gradient_descent(task4_f, 0.2,-4)[0], gradient_descent(task4_f, 0.2,-4)[1])
Hessian4_2 = hessian(task4_f, gradient_descent(task4_f, 0.2,-4)[0], gradient_descent(task4_f, 0.2,-4)[1])
# Gradient and Hessian for x_20 starting from either starting value

code.interact(local=locals())
