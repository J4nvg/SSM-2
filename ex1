import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patches import Circle

# 1.a)

# Wiener process
def wienerProcess(T,M):
    dt = T/M
    normDist = stats.norm(0,np.sqrt(dt))
    incrementsX = normDist.rvs(M)
    incrementsY = normDist.rvs(M)
    sbmX = np.append([0],np.cumsum(incrementsX))
    sbmY = np.append([0],np.cumsum(incrementsY))
    return sbmX, sbmY
"""
# plot
X, Y = wienerProcess(100,2000)
plt.figure()
plt.plot(X, Y)
plt.show()
"""


# 1.b)

"""
The expectation of tau is around 1.4 instead of 0.5 
"""

T = 100
M = 20000

# Wiener process returning hitting time
def wienerProcessHit(T,M,hit):
    dt = T/M
    normDist = stats.norm(0,np.sqrt(dt))
    incrementsX = normDist.rvs(M)
    incrementsY = normDist.rvs(M)
    
    distance = 0 # distance from center
    m = 0 # time the process hits "hit"
    x = np.cumsum(incrementsX)
    y = np.cumsum(incrementsY)
    for i in range(M):
        distance = np.sqrt(x**2 + y**2)[i]
        m = i
        if distance >= hit:
            break
    """
    # test
    sbmX = np.append([0],np.cumsum(incrementsX[0:m]))
    sbmY = np.append([0],np.cumsum(incrementsY[0:m]))
    circle = Circle((0,0), hit, fill=False, color="red")
    fig, ax = plt.subplots()
    ax.set_xlim((-1.5, 1.5))
    ax.set_ylim((-1.5, 1.5))
    fig = plt.plot(sbmX, sbmY)
    ax.add_patch(circle)
    plt.show()
    """
    return T/m # tau

# simulation
nrRuns = 100
sim = np.zeros(nrRuns) # tau values
for i in range(nrRuns):
    sim[i] = wienerProcessHit(T,M,1)
expTau = np.mean(sim)
print(expTau)


# 1.c)

n = 10

# function to determine radii
def r(i):
    return 4**(-1) + (4*10)**(i/n)

sim = np.zeros((n-3, nrRuns+1)) # tau's of i* = 2, 3, ... , n-2
for i in range(2,n-1):
    for j in range(nrRuns):
        sim[i-2, 0] = i
        tau = wienerProcessHit(T,M,r(i))
        sim[i-2, j+1] = tau
# print(sim)

expTable = np.zeros((n-3, 4))
simExp1 = np.zeros(n-3) # value of the first expectation in each simulation
simProb = np.zeros(n-3) # probability that the condition holds in each simualtion
simExp2 = np.zeros(n-3) # value of the second expectation in each simulation
for s in range(nrRuns): # simulation
    exp1True = [] # where the condition is true for 1st expectation
    exp2True = [] # where the condition is true for 2nd expectation
    for i in range(1, n-4): # i*
        if sim[i-1, s] < sim[i+1, s]:
            exp1True.append(sim[i-1, s])
        probCounter = 0 # when the condition hold for the probability
        if sim[i+1, s] < sim[i-1, s]:
            probCounter += 1
        if sim[i-1, s] > sim[i+1, s]:
            exp2True.append(sim[i+1, s])
    # calculating all asked expectations/probability in given simulation
    e1 = np.mean(exp1True)
    p = probCounter/(n-3)
    e2 = np.mean(exp2True)
    # storing them
    simExp1 = e1
    simProb = p
    simExp2 = e2
    
for i in range(n-3):
    expTable[i, 0] = i+2
    expTable[i, 1] = np.mean(simExp1)
    expTable[i, 2] = np.mean(simProb)
    expTable[i, 3] = np.mean(simExp2)
print(expTable)
