import numpy as np
import math
import matplotlib.pyplot as plt


def normal_C_P_N(P,N):   
    result = 0
    for i in range(N):
        result += math.comb((P-1), i)
    return result * 2

def bound_high_P(P,N):
    return (np.exp(1)*P/N)**(N)

def bound_low_P(P,N):
    return 2**(P)

N = 50
P_low = [bound_low_P(P,N) for P in range(1,51)]
P_high = [bound_high_P(P,N) for P in range(51,201)]
P_total = np.array(P_low + P_high)
real_values = np.array([normal_C_P_N(P,N) for P in range(1,201)])

difference = P_total - real_values
results = [(i + 1,val) for i, val in enumerate(zip(P_total, real_values))]
print("Below the difference between the bound and the real value is printed, where the bound is calculated differently for P smaller than N and bigger")

print(difference)

#As we can see within the print. The bound introduced for the P > N is very conservative!



