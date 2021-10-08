import numpy as np
import matplotlib.pyplot as plt
import math


def posterior(lam, r):
    return math.exp(-lam) * lam**r / math.factorial(r)


# N(ƛ; r-1, r-1)
def py1(lam, r):
    exponent = (-1/(2*(r-1))) * (lam-(r-1))**2
    gaussian = 1/math.sqrt(2*math.pi*(r-1)) * math.exp(exponent)
    return gaussian


# N(log(ƛ); log(r), 1/r)
def py2(lam, r):
    exponent = (-1/(2*(1/r))) * (np.log(lam)-np.log(r))**2
    gaussian = 1/math.sqrt(2*math.pi*(1/r)) * math.exp(exponent)
    return gaussian


x = np.linspace(0,20, 200)

r=2 #10
posteriors = np.array([posterior(i,r) for i in x])
py1_list = np.array([py1(i,r) for i in x])
py2_list = np.array([py2(i,r) for i in x])

# setting the axes at the centre
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.margins(0,0)

# plot the function
plt.plot(x,posteriors/np.sum(posteriors), 'r',label="p")
plt.plot(x,py1_list/np.sum(py1_list),'b', label="p1")
plt.plot(x,py2_list/np.sum(py2_list),'g', label="p2")
plt.legend(loc="upper right")

# show the plot
plt.show()
