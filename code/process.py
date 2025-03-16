import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# A circular semi-permeable barrier class to handle reflections.
class CSB:
    def __init__(self, x, y, r, lp, ln, s=1):
        self.x = x
        self.y = y
        self.r = r
        self.lp = lp
        self.ln = ln
        self.s = s
        l = self.ln if self.s == 1 else self.lp
        self.next_event = stats.expon.rvs(scale=1/l) if l != 0 else np.inf
        self.local_time = 0

    # Reflect or tunnel the process through the barrier and update local time
    def reflect(self, X, dt, eps=0.001):
        # Change coordinates to center the circle
        X -= (self.x, self.y)
        # Reflect process back to boundary when the process goes through
        if np.linalg.norm(X) * self.s > self.r * self.s:
            X *= self.r / np.linalg.norm(X)
        # Increment local time
        if np.abs(np.linalg.norm(X) - self.r) < eps:
            self.local_time += dt / (2 * eps)
            # Update Markov-chain barrier state.
            while self.local_time > self.next_event:
                self.s *= -1
                l = self.ln if self.s == 1 else self.lp
                self.next_event += stats.expon.rvs(scale=1/l) if l != 0 else np.inf

        # Go back to original coordinates
        X += (self.x, self.y)
    
    def draw(self, ax, color='b'):
        c = plt.Circle((self.x, self.y), self.r, color=color, fill=False)
        ax.add_patch(c)

# A 2D brownian motion process with duration T, M timesteps, initial location of W0 and barriers B
def process(T, M, W0, B):
    dt = T/M
    dW = stats.norm(0, np.sqrt(dt)).rvs((2, M))
    if not B: # Faster when no boundaries
        return np.cumsum(np.append([[W0[0]],[W0[1]]], dW, axis=1), axis=1)
    W = np.zeros((2, M+1))
    W[:, 0] = W0
    for i in range(M):
        W[:, i+1] = W[:, i] + dW[:, i]
        for b in B:
            b.reflect(W[:, i+1], dt, eps=2*dt)
    return W

# Runs the process until it hits some radius r, return the elapsed time
def hitting_time(M, W0, r, B=[]):
    s = 1 if np.linalg.norm(W0) < r else -1 # get original orientation
    W = process(1, M, W0, B) # simulate the process for a duration of 1
    i = np.nonzero(np.linalg.norm(W, axis=0) * s >= r * s)[0]
    if i.size > 0:
        return i[0] / M
    else:
        return 1 + hitting_time(M, W[:, -1], r, B)
    
def conf(X):
    return f"{np.mean(X):.3} \\pm {1.96 * np.std(X)/np.sqrt(X.size):.3}"