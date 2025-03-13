import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class CSB:
    def __init__(self, x, y, r, lp, ln, s=1):
        self.x = x
        self.y = y
        self.r = r
        self.lp = lp
        self.ln = ln
        self.s = s
        l = self.ln if self.s == 1 else self.lp
        self.next_event = stats.poisson.rvs(l) if l != np.inf else np.inf
        self.local_time = 0

    def reflect(self, X, dt, eps=0.001):
        oX = X
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
    
    def draw(self, ax, color='b'):
        c = plt.Circle((self.x, self.y), self.r, color=color, fill=False)
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
            b.reflect(X[:, i], dt, eps=2*dt)
    # plt.plot(X[0], X[1])
    # ax = plt.gca()
    # for b in B:
    #     b.draw(ax)
    # plt.show()
    return X


# -- 3B --
# N = 100
# XT = np.zeros(N)
# for i in range(N):
#     XT[i] = np.linalg.norm(process(100, 1000, (0, 0), CSB(0, 0, 1, 0, np.inf), CSB(0, 0, 1/2, 4, 1))[:, -1])

# eps = 10/N
# Eta = np.linspace(eps/2, 1-eps/2, N//10)
# I = np.zeros(N//10)
# for i, eta in enumerate(Eta):
#     I[i] = 1/(2 * eta * eps) * np.mean(np.abs(XT - eta) < eps / 2)
# plt.plot(Eta, I)
# plt.show()


# -- 3C --
# process(1, 100000, (0,0), CSB(0,0,1,0,np.inf), CSB(1/4, 0, 1/2, 2, 1/2), CSB(-1/2, 1/2, 1/8, 3, 1/3, s=-1), CSB(1/2, 1/4, 1/16, 1/4, 4, s=-1))
T = 100
M = 1000000
s = T/M
w = 0.002
B = CSB(0,0,1,0,np.inf), CSB(1/4, 0, 1/2, 2, 1/2), CSB(-1/2, 1/2, 1/8, 3, 1/3, s=-1), CSB(1/2, 1/4, 1/16, 1/4, 4, s=-1)
X = process(T, M, (0, 0), *B)
# H = np.histogram2d(x, y, bins=(2/w, 2/w), range=[[-1,1],[-1,1]])

plt.hist2d(X[0], X[1], bins=(int(2/w), int(2/w)), range=[[-1,1], [-1,1]])
ax = plt.gca()
for b in B:
    b.draw(ax, color='w')
plt.show()