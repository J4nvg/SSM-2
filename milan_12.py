import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def wiener_process(T, M, x=0, y=0):
    dt = T/M
    normDist = stats.norm(0, np.sqrt(dt))
    dX = normDist.rvs(M)
    dY = normDist.rvs(M)
    X = np.append([0], np.cumsum(dX)) + x
    Y = np.append([0], np.cumsum(dY)) + y
    return X, Y

X, Y = wiener_process(10, 1000)
# plt.plot(X, Y)
# plt.show()

def circle(M, *rs):
    ax = plt.gca()
    for r in rs:
        c = plt.Circle((0, 0), r, color='b', fill=False)
        ax.add_patch(c)


def hitting_time(dt, x=0, y=0):
    X, Y = wiener_process(1, int(1/dt))
    X += x
    Y += y
    i = np.argmax(X*X + Y*Y >= 1)
    # circle((0,0), 1)
    # plt.plot(X[0:i], Y[0:i])
    # plt.show()
    return i * dt if i > 0 else 1 - dt + hitting_time(dt, X[-1], Y[-1])

# N = 10000
# T = [hitting_time(0.0001) for i in range(N)]
# m = np.mean(T)
# h = 1.96 * np.std(T) / np.sqrt(N)
# print(m - h, m, m + h)

# C
def first(A):
    I = np.nonzero(A)[0]
    return I[0] if I.size > 0 else np.inf

def hitting_time2(M, r1, r2, x=0, y=0):
    dt = 1/M
    X, Y = wiener_process(1, M)
    X += x
    Y += y
    i = min(first(X*X + Y*Y <= r1*r1), first(X*X + Y*Y >= r2*r2))
    # plt.plot(X[:i+1 if i != np.inf else -1], Y[:i+1 if i != np.inf else -1])
    # circle((0, 0), r1, r2)
    # plt.show()
    if i == np.inf:
        t, h1 = hitting_time2(M, r1, r2, X[-1], Y[-1])
        return t + 1, h1
    else:
        return i*dt, X[i]*X[i] + Y[i]*Y[i] <= r1*r1
    

# n = 10
# N = 1000
# r = lambda i: 40 ** (i / n) / 4
# for i in range(2, n-1):
#     T = np.zeros(N)
#     H1 = np.zeros(N)
#     for k in range(N):
#         T[k], H1[k] = hitting_time2(1000, r(i-1), r(i+1), r(i), 0)
#     print(f"n={i} &", np.mean(T[H1 == 0]), "&", np.mean(H1), "&", np.mean(T[H1 == 1]), "\\\\")


# 2A
def reflected_brownian_motion(T, M, r1, r2, x=0, y=0):
    dt = T/M
    normDist = stats.norm(0, np.sqrt(dt))
    dX = normDist.rvs((2, M))
    X = np.zeros((2, M+1))
    X[:, 0] = [x, y]
    for i in range(1, M+1):
        X[:, i] = X[:, i-1] + dX[:, i-1]
        if np.linalg.norm(X[:, i]) < r1:
            X[:, i] *= r1 / np.linalg.norm(X[:, i])
        elif np.linalg.norm(X[:, i]) > r2:
            X[:, i] *= r2 / np.linalg.norm(X[:, i])
    # plt.plot(X[0], X[1])
    # circle((0, 0), r1, r2)
    # plt.show()
    return X

def hitting_r2(M, r1, r2, x=0, y=0):
    X = reflected_brownian_motion(1, M, r1, np.inf, x, y)
    i = first(np.linalg.norm(X, axis=0) >= r2)
    if i == np.inf:
        return 1 + hitting_r2(M, r1, r2, *X[:, -1])
    plt.plot(X[0, :i], X[1, :i])
    circle((0, 0), r1, r2)
    plt.show()
    return i / M

print(hitting_r2(10000, 1/2, 1, 3/4, 0))

def hitting_r1(M, r1, r2, x=0, y=0):
    X = reflected_brownian_motion(1, M, 0, r2, x, y)
    i = first(np.linalg.norm(X, axis=0) <= r1)
    if i == np.inf:
        return 1 + hitting_r2(M, r1, r2, *X[:, -1])
    plt.plot(X[0, :i], X[1, :i])
    circle((0, 0), r1, r2)
    plt.show()
    return i / M

print(hitting_r1(10000, 1/2, 1, 3/4, 0))

