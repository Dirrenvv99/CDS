import numpy as np
import math
import matplotlib.pyplot as plt


def normal_C_P_N(P,N):   
    return np.sum(np.array([math.comb((P-1), i) for i in range(N)])) * 2

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
print(np.sum(np.array([math.comb((32-1), i) for i in range(N)])))

#plt.plot(np.array([P for P in range(1, 201)]), P_total, label = "Bound values")
#plt.plot(np.array([P for P in range(1,201)]), real_values, label = "Numerical values")
#plt.legend()
#plt.xlim([1,50])
#plt.ylim([1,1000000])
#plt.show()



