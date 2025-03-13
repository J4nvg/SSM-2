import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class CSB:
    def __init__(self, x, y, r, lp, ln):
        self.x = x
        self.y = y
        self.r = r
        self.lp = lp
        self.ln = ln
        self.s = 1
        self.next_event = stats.poisson.rvs(self.ln) if self.ln != np.inf else np.inf
        self.local_time = 0

    def reflect(self, X, dt, eps=0.001):
        X -= (self.x, self.y)
        if np.linalg.norm(X) * self.s > self.r * self.s:
            X *= self.r / np.linalg.norm(X)
        if np.abs(np.linalg.norm(X) - self.r) < eps:
            self.local_time += dt / (2 * eps)
            while self.local_time > self.next_event:
                self.next_event += stats.poisson.rvs(self.ln if self.s == 1 else self.lp)
                self.s *= -1
        X += (self.x, self.y)
        return X
    
    def draw(self, ax):
        c = plt.Circle((self.x, self.y), self.r, color='b', fill=False)
        ax.add_patch(c)

def process(T, M, X0, *B):
    X = np.zeros((2, M+1))
    X[:, 0] = X0
    dt = T/M
    normDist = stats.norm(0, np.sqrt(dt))
    dX = normDist.rvs((2, M))
    for i in range(1, M+1):
        X[:, i] = X[:, i-1] + dX[:, i-1]
        for b in B:
            b.reflect(X[:, i], dt)
    plt.plot(X[0], X[1])
    ax = plt.gca()
    for b in B:
        b.draw(ax)
        print(b.local_time)
    plt.show()
    return X

process(2, 100000, (0, 0), CSB(0, 0, 1, 0, np.inf), CSB(0, 0, 1/2, 4, 1))

