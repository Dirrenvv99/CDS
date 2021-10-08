import numpy as np


even = np.array([0,2,4,6,8,10])
N=10

for u in range(N+1):
    prob = 0
    for x in even:
        fu = u/N
        p = fu**x * (1-fu)**(N-x)

        total = 0
        for u2 in range(N+1):
            fu2 = u2/N
            total += fu2**x * (1-fu2)**(N-x)
 
        prob += p / total

    print(prob)
   