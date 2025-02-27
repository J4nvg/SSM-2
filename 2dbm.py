import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from dist.Distribution import Distribution

nrRuns = 100

def twodimensionalBM(mu, sigma, T, M):
    # hitting_times = np.zeros(nrRuns)
    x, y = np.zeros(M), np.zeros(M)
    dt = T / M
    normDist = Distribution(stats.norm(mu * dt, sigma * np.sqrt(dt)))
    dX = normDist.rvs(M)
    dY = normDist.rvs(M)
    for i in range(1,M):
        x[i] = x[i-1] + dX[i]
        y[i] = y[i-1] + dY[i]
    bm = np.column_stack((x, y))
    return bm

mu = 1
sigma = 0.5

T = 5
M = 1000

bm = twodimensionalBM(mu, sigma, T, M)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(0,0,'ro')
plt.plot(bm[:, 0], bm[:, 1])
plt.title("2D Brownian Motion")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()