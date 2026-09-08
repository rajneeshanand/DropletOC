# for Reducing Cost-Compatibility Verification of Control Lyapunov
#Functions to a Single Inequality--Aircraft problem

import numpy as np
from scipy.optimize import minimize

J = np.array([1., 2., 3.])
sigma = (J[2]-J[1])/J[0] + (J[0]-J[2])/J[1] + (J[1]-J[0])/J[2]  # = 1/3
r = 10.0

def neg_ratio(p):
    th, ph = p
    w = r*np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
    LfV = -sigma*w[0]*w[1]*w[2]
    A = 0.25*np.sum(w**2/J**2)
    return -LfV/A

best = None
for _ in range(300):
    x0 = np.random.rand(2)*[np.pi, 2*np.pi]
    res = minimize(neg_ratio, x0)
    if best is None or res.fun < best.fun:
        best = res

th, ph = best.x
w = r*np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
print("alpha* =", -best.fun)
print("omega  =", np.round(w, 2))