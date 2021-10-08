import numpy as np
import scipy.integrate as integrate

def func(m):
    """Function that we integrated numerically"""
    return (1 + 0.3 * m) * (1 + 0.5 * m) * (1 + 0.7 * m) * (1 + 0.8 * m) * (1 + 0.9 * m)

print(integrate.quad(func, -1, 1)[0]/64)
