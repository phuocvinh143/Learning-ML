import math
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt

def grad(x):
    return 2*x + 10*np.cos(x)

def has_converged(theta_new):
    return np.linalg.norm(grad(theta_new))/len(theta_new) < 1e-3

def GD_momentum(theta_init, eta, gamma=.9):
    theta = [theta_init]
    v_old = np.zeros_like(theta_init)
    for it in range(100):
        v_new = gamma*v_old + eta*grad(theta[-1])
        theta_new = theta[-1] - v_new
        if has_converged(theta_new):
            break
        theta.append(theta_new)
        v_old = v_new
    return theta

x = -4
print(GD_momentum(x, 0.1))