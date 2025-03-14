from process import *

# 2B
r2 = 1
N = 100
for r1 in [1/8, 1/4, 1/2]:
    X = np.linspace(r1, r2, 20)
    T = np.zeros((20, N))
    for i in range(20):
        T[i] = [hitting_time(1000, [X[i], 0], 1, [CSB(0, 0, r1, np.inf, 0, -1)]) for _ in range(N)]
    plt.plot(X, np.mean(T, axis=1))
    plt.plot(X, np.mean(T, axis=1) - 1.96 * np.std(T, axis=1) / np.sqrt(N))
    plt.plot(X, np.mean(T, axis=1) + 1.96 * np.std(T, axis=1) / np.sqrt(N))
    plt.plot(X, 0.5*(1-X**2 + r1**2 * np.log(X**2)))
    plt.show()

# 2C
N = 100
for x0 in [0.6, 0.7, 0.8, 0.9]:
    T = np.zeros((4, N))
    for i, r1 in enumerate([0.1, 0.2, 0.3, 0.4]):
        T[i] = [hitting_time(1000, [x0, 0], r1, [CSB(0, 0, 1, 0, np.inf)]) for _ in range(N)]
    print(f"x_0 = {x0} & {" & ".join(conf(T[i]) for i in range(4))} \\\\")