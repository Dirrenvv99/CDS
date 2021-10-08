import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math

def epsilon(N,P, delta):
    if P <= N:
        return np.sqrt(8 * ( (2*P + 2)* np.log(2) - np.log(delta))/P)
    else:
        return np.sqrt((8 * ( N *  np.log((2*P)/N) + N - np.log(delta/4)))/P)


def fit_func(P, a):
    return P**(a)


num_patterns = []
for N in range(10,51,10):
    eps_data = [epsilon(N,P,0.01) for P in range(1,2*N + N)]
    P_data = [P for P in range(1,2*N +N)]
    print(P_data)
    print(eps_data)

    popt, pcov = curve_fit(fit_func, P_data, eps_data)
     
    num_patterns.append(math.ceil((0.1)**(1/popt)))

print(num_patterns)




