import numpy as np
import math

def likelihood_fair(data):
    k = len(data)
    n = np.sum(data)
    denom = 1
    for i in data:
        denom *= math.factorial(i)
    for _ in range(n):
        denom *= k
    return math.factorial(n)/denom

def likelihood_H_1(data):
    k = len(data)
    n = np.sum(data)
    return math.factorial(n) * math.factorial(k-1)/math.factorial(n + k -1)

data1 = (3,3,2,2,9,11)
data2 = (5,5,5,5,5,5)
denom_data1 = likelihood_fair(data1) + likelihood_H_1(data1)
denom_data2 = likelihood_fair(data2) + likelihood_H_1(data2)

print("Posterior for data1 H_0:", likelihood_fair(data1)/(denom_data1))
print("Posterior for data1 H_1:", likelihood_H_1(data1)/(denom_data1))
print("Posterior for data2 H_0:", likelihood_fair(data2)/(denom_data2))
print("Posterior for data2 H_1:", likelihood_H_1(data2)/(denom_data2))



