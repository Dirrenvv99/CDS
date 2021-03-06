import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math


def epsilon(N,P, delta):
    '''The epsilon function, for its derivation see the pdf'''
    if P <= N:
        return np.sqrt(8 * ( (2*P + 2)* np.log(2) - np.log(delta))/P)
    else:
        return np.sqrt((8 * ( N *  np.log((2*P)/N) + N - np.log(delta/4)))/P)


def fit_func(P, a):
    '''Function that is fitted, in order to determine P dependence of epsilon for given N'''
    return P**(a)

N = 10
eps_data = [epsilon(N,P,0.01) for P in range(1,450000)]
P_data = [P for P in range(1,450000)]
#Fitting the P dependence
popt, pcov = curve_fit(fit_func, P_data, eps_data)
#Determining Patterns needed for epsilon = 0.1
patterns_10 = math.ceil((0.1)**(1/popt[0]))
print("We find for N = ", N, " that we need: ", patterns_10 , "patterns, to have a error of: ", epsilon(N,patterns_10, 0.01))

#Determining the patterns needed for the other N by using the linearity
error_01 = [i * patterns_10 for i in range(1,6)]
print("The number of patterns needed for an error of 0.1 is: ")
print([((idx + 1) * 10, val) for idx, val in enumerate(error_01)])




