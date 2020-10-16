import math
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt

def grad(x):
    return 2*x + 5*np.cos(x)

def cost(x):
    return x**2 + 5*np.sin(x)

def myGD1(eta, x0):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, it)

(x1, it1) = myGD1(.1, 5)
(x2, it2) = myGD1(.1, -5)

print('x = ' + str(x1[-1]), end = ' ')
print("after " + str(it1) + " steps")
print('x = ' + str(x2[-1]), end = ' ')
print("after " + str(it2) + " steps")