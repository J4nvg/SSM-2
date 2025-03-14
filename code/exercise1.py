from process import *

# 1B
N = 1000
T = np.zeros(N) # hitting times
for i in range(T.size):
    T[i] = hitting_time(10000, (0,0), 1)

print(conf(T))

# 1C
# Simulate until the process hits either the inner (1) or outer (2) circle, 
# return hitting time and boolean that the outer circle got hit
def hitting_time2(M, r1, r2, W0):
    W = process(1, M, W0, B=[])
    norm = np.linalg.norm(W, axis=0)
    i = np.nonzero((norm <= r1) | (norm >= r2))[0]
    if i.size > 0:
        return i[0] / M, norm[i[0]] >= r2
    else:
        T, I = hitting_time2(M, r1, r2, W[:, -1])
        return T+1, I
        

n = 10
N = 1000
r = lambda i: 40 ** (i / n) / 4 # circle radii
for i in range(2, n - 1):
    T = np.zeros(N) # hitting times
    I = np.zeros(N) # indicators that the smaller circle was hit
    for k in range(N):
        T[k], I[k] = hitting_time2(1000, r(i-1), r(i+1), [r(i), 0])
    print(f"{i} & {conf(T[I == 0])} & {conf(I)} & {conf(T[I == 1])}\\\\")