from process import *

# 3B
N = 50
XT = np.zeros(N)
B = CSB(0, 0, 1, 0, np.inf), CSB(0, 0, 1/2, 4, 1)


for i in range(N):
    print(i)
    XT[i] = np.linalg.norm(process(100, 5000, (0, 0), B)[:, -1])

eps = 0.2
Eta = np.linspace(eps/2, 1, 50)
P = np.zeros(Eta.size) # scaled probabilities
H = np.zeros(Eta.size) # confidence band halfwidths
for i, eta in enumerate(Eta):
    p = np.mean(np.abs(XT - eta) < eps / 2)
    P[i] = 1/(2 * eta * eps) * p
    H[i] = 1/(2 * eta * eps) * 1.96 * np.sqrt(p * (1-p) / N)

plt.figure(dpi=700)
plt.plot(Eta, P)
plt.plot(Eta, P-H, color='r')
plt.plot(Eta, P+H, color='r')
plt.xlabel("η")
plt.savefig("norms_plot.png")

# 3C
T = 100
M = 1000000
s = T/M
w = 0.005
B = CSB(0,0,1,0,np.inf), CSB(1/4, 0, 1/2, 2, 1/2), CSB(-1/2, 1/2, 1/8, 3, 1/3, s=-1), CSB(1/2, 1/4, 1/16, 1/4, 4, s=-1)
X = process(T, M, (0, 0), B)
plt.figure(dpi=700)
plt.hist2d(X[0], X[1], bins=(int(2/w), int(2/w)), range=[[-1,1], [-1,1]])
ax = plt.gca()

plt.colorbar()
plt.title("Binned counts for a simulation with Semipermeable Barriers")
plt.savefig("simulation_binned_counts.png")